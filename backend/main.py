"""
GreenRAG Backend — FastAPI entrypoint.
RAG pipeline for sustainability assessment of cloud AI infrastructure.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="GreenRAG API",
    description="RAG-based sustainability assessment for cloud AI infrastructure.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess")
def assess(payload: dict):
    """
    Accepts a natural-language infrastructure description and returns
    sustainability recommendations grounded in retrieved research.
    Retrieval + generation pipeline to be wired in next.
    """
    return {"recommendations": [], "sources": [], "status": "pipeline not yet implemented"}
