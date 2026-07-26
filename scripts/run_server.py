#!/usr/bin/env python3
"""
Run the FastAPI analytics server
Usage: python scripts/run_server.py
"""

import uvicorn
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start the FastAPI server."""
    logger.info("="*60)
    logger.info("STARTING AI PRODUCT INSIGHTS DASHBOARD API")
    logger.info("="*60)
    logger.info("")
    logger.info("📊 Server: http://localhost:8000")
    logger.info("📖 API Docs: http://localhost:8000/docs")
    logger.info("🔄 ReDoc: http://localhost:8000/redoc")
    logger.info("")
    logger.info("Available Endpoints:")
    logger.info("  GET /api/v1/trends - Top clusters by activity")
    logger.info("  GET /api/v1/complaints/top - Top complaints by severity")
    logger.info("  GET /api/v1/dashboard/summary - Dashboard overview")
    logger.info("  GET /api/v1/alerts/urgent - Critical issues")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*60 + "\n")
    
    uvicorn.run(
        "ai_product_insights_dashboard.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n✓ Server stopped")
        sys.exit(0)
