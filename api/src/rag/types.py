from pydantic import BaseModel, Field
from typing import List, Optional, Literal

RagMode = Literal["semantic", "semantic_rerank", "bm25", "hybrid", "no_rag"]

class QuestionRequest(BaseModel):
    question: str
    mode: RagMode = "hybrid"
    top_k: int = 5
    alpha: float = 0.5
    rerank_property: str = "chunk"

class DocRef(BaseModel):
    title: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None
    link: Optional[str] = None
    chunk: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    references: List[str]
    contexts: List[DocRef]


