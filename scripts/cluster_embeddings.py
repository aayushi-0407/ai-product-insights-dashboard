#!/usr/bin/env python3
"""
Phase 2a: Clustering & Labeling
- Load embeddings from processed chunks
- Run HDBSCAN clustering
- Label clusters with LLM (Groq - free)
- Save results to PostgreSQL
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import hdbscan
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# See ai_product_insights_dashboard.app for why this runs before Groq/Postgres calls.
import truststore

truststore.inject_into_ssl()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Try importing Groq (optional for labeling)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("⚠ Groq not installed. Cluster labeling will be skipped.")

# Try importing PostgreSQL (optional for persistence)
try:
    from sqlalchemy import create_engine, text
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("⚠ SQLAlchemy not installed. PostgreSQL persistence will be skipped.")


def load_embeddings_from_jsonl(jsonl_path: str) -> Tuple[List[str], np.ndarray]:
    """Load embeddings from processed review chunks JSONL file."""
    logger.info(f"Loading embeddings from {jsonl_path}...")
    
    chunk_ids = []
    embeddings = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunk_ids.append(chunk['chunk_id'])
            embeddings.append(chunk['embedding'])
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    logger.info(f"✓ Loaded {len(embeddings)} embeddings ({embeddings_array.shape[1]} dimensions)")
    
    return chunk_ids, embeddings_array


def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 3) -> Tuple[np.ndarray, int]:
    """
    Cluster embeddings using HDBSCAN.
    Returns cluster labels (where -1 = noise/outliers) and number of clusters.
    """
    logger.info(f"Running HDBSCAN clustering (min_cluster_size={min_cluster_size})...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise cluster
    n_noise = sum(labels == -1)

    # A ten-review demo corpus has no stable density regions, so HDBSCAN
    # correctly returns only noise. Preserve HDBSCAN as the primary approach,
    # then make a small, deterministic KMeans fallback available for demos.
    if n_clusters == 0 and len(embeddings) >= 2:
        fallback_clusters = min(3, max(2, len(embeddings) // 4))
        logger.warning(
            "HDBSCAN found only noise in this small corpus; using KMeans "
            "fallback with %s clusters.", fallback_clusters
        )
        labels = KMeans(n_clusters=fallback_clusters, random_state=42, n_init=10).fit_predict(embeddings)
        n_clusters = fallback_clusters
        n_noise = 0
    
    logger.info(f"✓ Found {n_clusters} clusters with {n_noise} noise points")
    
    return labels, n_clusters


def _local_cluster_label(sample_texts: List[str], cluster_id: int) -> str:
    """Create a useful offline label when no Groq key is available."""
    combined = " ".join(sample_texts).lower()
    themes = [
        ("Authentication issues", ("login", "authentication", "password")),
        ("Installation problems", ("install", "setup", "installation")),
        ("Stability & crashes", ("crash", "broken", "fail", "error")),
        ("Customer support", ("support", "customer service")),
        ("Product experience", ("great", "excellent", "amazing", "love", "recommend")),
        ("Pricing concerns", ("price", "worth", "cost")),
    ]
    matches = [
        (sum(combined.count(keyword) for keyword in keywords), label)
        for label, keywords in themes
    ]
    match_count, label = max(matches, key=lambda item: item[0])
    if match_count:
        return label
    return f"Feedback theme {cluster_id + 1}"


def get_cluster_samples(chunk_ids: List[str], embeddings: np.ndarray, labels: np.ndarray, cluster_id: int, n_samples: int = 3) -> List[str]:
    """Get sample chunk texts/ids from a cluster for LLM labeling."""
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    
    # Get closest to centroid
    if len(cluster_indices) > 0:
        centroid = embeddings[cluster_indices].mean(axis=0)
        distances = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
        closest_indices = cluster_indices[np.argsort(distances)[:n_samples]]
        return [chunk_ids[i] for i in closest_indices]
    
    return []


def label_clusters_with_groq(chunk_ids: List[str], embeddings: np.ndarray, labels: np.ndarray, 
                              jsonl_path: str, n_clusters: int) -> Dict[int, str]:
    """Generate cluster labels using Groq LLM."""
    # Load chunk texts once for either local or LLM labels.
    chunks_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks_data.append(json.loads(line))

    def cluster_texts(cluster_id: int) -> List[str]:
        """Use row positions: sample datasets can repeat a review/chunk id."""
        return [chunks_data[index]['text'] for index in np.where(labels == cluster_id)[0]]

    if not GROQ_AVAILABLE:
        logger.warning("⚠ Groq not available. Generating local labels...")
        return {i: _local_cluster_label(cluster_texts(i), i) for i in range(n_clusters)}
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.warning("⚠ GROQ_API_KEY not set. Generating local labels...")
        return {i: _local_cluster_label(cluster_texts(i), i) for i in range(n_clusters)}
    
    logger.info("Generating cluster labels with Groq (free tier)...")
    
    client = Groq(api_key=groq_key)
    labels_dict = {}
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        sample_texts = [chunks_data[index]['text'] for index in cluster_indices[:3]]
        
        prompt = f"""Analyze these 3 review snippets from a product review cluster and generate a short, actionable label (3-5 words):

Snippet 1: {sample_texts[0]}
Snippet 2: {sample_texts[1] if len(sample_texts) > 1 else "N/A"}
Snippet 3: {sample_texts[2] if len(sample_texts) > 2 else "N/A"}

Generate a concise label that captures the main theme (e.g., "Performance Issues", "UI Improvements", "Bug Reports", "Feature Requests"). 
Return ONLY the label, no explanation."""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            label = response.choices[0].message.content.strip()
            labels_dict[cluster_id] = label
            logger.info(f"  Cluster {cluster_id}: {label}")
        except Exception as e:
            logger.warning(f"  Cluster {cluster_id}: Failed to label ({str(e)}), using generic label")
            labels_dict[cluster_id] = _local_cluster_label(sample_texts, cluster_id)
    
    return labels_dict


def save_to_postgres(chunk_ids: List[str], labels: np.ndarray, label_names: Dict[int, str],
                     jsonl_path: str) -> bool:
    """Save cluster assignments to PostgreSQL."""
    if not POSTGRES_AVAILABLE:
        logger.warning("⚠ SQLAlchemy not available. Skipping PostgreSQL persistence.")
        return False

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("⚠ DATABASE_URL not set. Skipping PostgreSQL persistence.")
        return False

    try:
        from ai_product_insights_dashboard.storage.postgres_store import update_cluster_assignments

        logger.info("Saving cluster assignments to PostgreSQL...")

        chunks_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks_data.append(json.loads(line))

        assignments = {}
        for i, chunk in enumerate(chunks_data):
            cluster_id = int(labels[i]) if labels[i] != -1 else None
            cluster_label = label_names.get(cluster_id, "Noise") if cluster_id is not None else "Noise"
            assignments[chunk['review_id']] = (cluster_id, cluster_label)

        updated = update_cluster_assignments(assignments)
        logger.info(f"✓ Saved cluster assignments for {updated} reviews")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save to PostgreSQL: {e}")
        return False


def save_clusters_to_jsonl(output_path: str, chunk_ids: List[str], labels: np.ndarray, 
                           label_names: Dict[int, str], jsonl_path: str):
    """Save clustering results alongside original data."""
    logger.info(f"Saving clustering results to {output_path}...")
    
    chunks_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks_data.append(json.loads(line))
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for i, chunk in enumerate(chunks_data):
            cluster_id = int(labels[i]) if labels[i] != -1 else None
            cluster_label = label_names.get(cluster_id, "Noise") if cluster_id is not None else "Noise"
            
            chunk['cluster_id'] = cluster_id
            cluster_label = label_names.get(cluster_id, "Noise") if cluster_id is not None else "Noise"
            chunk['cluster_label'] = cluster_label
            
            out_f.write(json.dumps(chunk) + '\n')
    
    logger.info(f"✓ Saved {len(chunks_data)} clustered chunks")


def print_cluster_summary(labels: np.ndarray, label_names: Dict[int, str]):
    """Print summary statistics about clusters."""
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    for label, count in sorted(zip(unique_labels, counts), key=lambda x: x[1], reverse=True):
        if label == -1:
            print(f"Noise/Outliers: {count} chunks")
        else:
            label_name = label_names.get(label, f"Cluster {label}")
            print(f"  {label_name}: {count} chunks")
    
    print("="*60 + "\n")


def main():
    """Main clustering pipeline."""
    print("\n" + "="*60)
    print("PHASE 2A: CLUSTERING & LABELING")
    print("="*60 + "\n")
    
    # Paths
    processed_chunks_path = Path("data/processed/review_chunks.jsonl")
    output_path = Path("data/processed/clustered_chunks.jsonl")
    
    if not processed_chunks_path.exists():
        logger.error(f"❌ {processed_chunks_path} not found. Run ingest_reviews.py first.")
        return
    
    # Load embeddings
    chunk_ids, embeddings = load_embeddings_from_jsonl(str(processed_chunks_path))
    
    # Cluster
    labels, n_clusters = cluster_embeddings(embeddings, min_cluster_size=5)
    
    # Label clusters (with or without LLM)
    label_names = label_clusters_with_groq(chunk_ids, embeddings, labels, str(processed_chunks_path), n_clusters)
    
    # Save results
    save_clusters_to_jsonl(str(output_path), chunk_ids, labels, label_names, str(processed_chunks_path))
    save_to_postgres(chunk_ids, labels, label_names, str(processed_chunks_path))
    
    # Summary
    print_cluster_summary(labels, label_names)
    logger.info("✓ Clustering pipeline complete!")


if __name__ == "__main__":
    main()
