"""
FastAPI Application - AI Product Insights Dashboard
Entry point for the analytics API server
"""

# Must run before any other module opens an HTTPS connection (Groq, Postgres,
# Pinecone, Hugging Face): managed Windows environments intercept TLS with a
# proxy certificate that isn't in the certifi bundle `httpx`/`requests` use
# by default, so outbound calls otherwise fail with CERTIFICATE_VERIFY_FAILED.
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import API routes
from .api.routes import router as analytics_router

# Initialize FastAPI app
app = FastAPI(
    title="AI Product Insights Dashboard",
    description="Analytics API for product review clustering & insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(analytics_router)


# ===== HEALTH CHECK =====

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "AI Product Insights Dashboard"
    }


# ===== ROOT ENDPOINT =====

@app.get("/")
async def root():
    """Serve the PM insights workspace."""
    static_file = Path(__file__).resolve().parents[2] / "static" / "index.html"
    return FileResponse(static_file)


# ===== STARTUP & SHUTDOWN =====

@app.on_event("startup")
async def startup_event():
    logger.info("✓ AI Product Insights Dashboard API starting...")
    logger.info("📊 Analytics endpoints available at /api/v1/")
    logger.info("📖 API docs available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("✓ AI Product Insights Dashboard API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ai_product_insights_dashboard.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
