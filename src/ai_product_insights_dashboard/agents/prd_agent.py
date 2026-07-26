"""PRD Agent: Generate Product Requirements Document from review insights."""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from ..api.routes import get_dashboard_summary, get_top_complaints


class PRDAgent:
    """Agent for generating PRD from review insights."""
    
    def __init__(self):
        self.name = "PRD Generation Agent"
        self.version = "1.0"
        self.groq_key = os.getenv("GROQ_API_KEY") if GROQ_AVAILABLE else None
    
    async def process(self, template: str = "standard", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate PRD from review insights.
        
        Args:
            template: PRD template style ("standard", "lean", "executive")
            metadata: Optional context
        
        Returns:
            Agent response with PRD content
        """
        logger.info(f"PRD Agent generating {template} PRD")
        
        try:
            # Collect dashboard data (routes.py returns Pydantic models, not dicts)
            summary = (await get_dashboard_summary()).model_dump()
            complaints = [c.model_dump() for c in await get_top_complaints(limit=5)]
            
            # Generate PRD
            prd = self._generate_prd(summary, complaints, template)
            
            return {
                "status": "success",
                "agent": self.name,
                "template": template,
                "prd": prd,
                "metadata": {
                    "total_reviews": summary.get("total_reviews", 0),
                    "clusters_analyzed": summary.get("total_clusters", 0),
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"PRD Agent error: {e}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e)
            }
    
    def _generate_prd(self, summary: Dict, complaints: List[Dict], template: str) -> str:
        """Generate PRD content."""
        if self.groq_key and GROQ_AVAILABLE:
            return self._generate_prd_with_llm(summary, complaints, template)
        else:
            return self._generate_prd_template(summary, complaints, template)
    
    def _generate_prd_with_llm(self, summary: Dict, complaints: List[Dict], template: str) -> str:
        """Use Groq to generate PRD."""
        try:
            client = Groq(api_key=self.groq_key)
            
            # Build context
            top_issues = "\n".join([
                f"- {c.get('cluster_label', 'Unknown')}: {c.get('complaint_count', 0)} mentions (severity: {c.get('severity_score', 0):.0f}/100)"
                for c in complaints[:5]
            ])
            
            prompt = f"""Generate a {template} Product Requirements Document (PRD) based on these review insights:

Dataset Summary:
- Total Reviews: {summary.get('total_reviews', 0)}
- Average Rating: {summary.get('avg_rating', 0):.1f}/5
- Negative Review %: {summary.get('negative_review_percentage', 0):.1f}%

Top Customer Issues:
{top_issues}

Format the PRD with:
1. Executive Summary
2. Key Problem Areas (from reviews)
3. Feature Requests
4. Quality Improvements
5. Priority Ranking

Be specific and actionable based on review data."""
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.warning(f"LLM PRD generation failed: {e}, using template")
            return self._generate_prd_template(summary, complaints, template)
    
    def _generate_prd_template(self, summary: Dict, complaints: List[Dict], template: str) -> str:
        """Generate PRD from template."""
        avg_rating = summary.get('avg_rating', 0)
        total_reviews = summary.get('total_reviews', 0)
        
        prd = f"""# PRODUCT REQUIREMENTS DOCUMENT

## Executive Summary

Based on analysis of {total_reviews} customer reviews, the product has an average rating of {avg_rating:.1f}/5.
This PRD outlines improvements needed to address the top customer pain points.

## Current State
- Total Reviews Analyzed: {total_reviews}
- Average Customer Rating: {avg_rating:.1f}/5.0
- Negative Feedback %: {summary.get('negative_review_percentage', 0):.1f}%
- Active Clusters: {summary.get('total_clusters', 0)}

## Top Customer Issues (Priority Order)
"""
        
        for i, complaint in enumerate(complaints[:5], 1):
            prd += f"""
### {i}. {complaint.get('cluster_label', 'Issue')}
- Mentions: {complaint.get('complaint_count', 0)}
- Severity Score: {complaint.get('severity_score', 0):.0f}/100
- Average Rating for This Issue: {complaint.get('avg_rating', 0):.1f}/5

"""
        
        prd += """## Recommended Actions

### High Priority (Implement ASAP)
- Address top 2 issues above
- Create test cases for reported bugs
- Setup customer feedback loop

### Medium Priority (Next Sprint)
- Implement feature requests from clusters 3-5
- Performance optimization
- UI/UX improvements

### Low Priority (Backlog)
- Minor enhancements
- Documentation improvements

## Success Metrics
- Reduce negative feedback rate by 25%
- Increase average rating to 4.5+
- Resolve top 3 complaint categories

## Timeline
- Phase 1 (High Priority): 2 weeks
- Phase 2 (Medium Priority): 4 weeks
- Phase 3 (Low Priority): Ongoing

---
Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return prd


def prd_agent():
    """Factory function for PRD agent."""
    return PRDAgent()
