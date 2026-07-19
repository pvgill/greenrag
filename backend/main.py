"""
GreenRAG Backend — FastAPI entrypoint.
RAG pipeline for sustainability assessment of cloud AI infrastructure.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="GreenRAG API",
    description="RAG-based sustainability assessment for cloud AI infrastructure.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssessRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=10,
        description="Natural-language description of the cloud AI infrastructure to assess.",
        examples=["16x A100 GPUs in us-east-1 running continuous LLM inference at full utilization."],
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of passages to retrieve.")


class RetrievedPassage(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class AssessResponse(BaseModel):
    query: str
    sources: list[RetrievedPassage]
    status: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index():
    """Build (or rebuild) the ChromaDB index from the corpus directory."""
    from rag.pipeline import build_index

    n = build_index()
    return {"status": "ok", "chunks_indexed": n}


@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest):
    """
    Accept a natural-language infrastructure description and return the
    top-k most relevant research passages ranked by cosine similarity.
    """
    from rag.pipeline import retrieve

    try:
        passages = retrieve(request.description, k=request.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval failed — ensure the index has been built via POST /index. Error: {exc}",
        ) from exc

    if not passages:
        raise HTTPException(
            status_code=404,
            detail="No passages retrieved. The index may be empty — call POST /index first.",
        )

    return AssessResponse(
        query=request.description,
        sources=[RetrievedPassage(**p) for p in passages],
        status="ok",
    )
