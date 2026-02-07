"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.analyze import router as analyze_router

# Setup logging
setup_logging("INFO" if not settings.debug else "DEBUG")
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI text detector using perplexity, burstiness, and repetition metrics",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
# For Phase 1: Restrict this to the GitHub Pages URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://meetp2022.github.io",
        "https://www.aichecking.me",
        "https://aichecking.me"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze_router)

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML (when running as monolith)."""
        return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/health")
async def root_health():
    """System health check."""
    return {"status": "ok", "app": settings.app_name}


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Model: {settings.model_name}")
    
    # Preload model (commented out to avoid blocking server startup)
    # from app.models.gpt2_loader import gpt2_loader
    # gpt2_loader.load()
    # logger.info("Model preloaded successfully")
    logger.info("Model will be loaded on the first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
