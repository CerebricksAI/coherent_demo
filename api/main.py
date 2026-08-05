import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.jobs import router as jobs_router

load_dotenv()

app = FastAPI(
    title="QSAID Work Instructions API",
    description="Upload factory videos, generate work instructions, stream results over WebSocket.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)


@app.get("/")
async def root():
    return {
        "service": "qsaid-wi",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "api": "/api/v1/jobs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "qsaid-wi", "version": "1.0.0"}
