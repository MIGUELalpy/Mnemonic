"""
Secondary node service for Mnemonic.
Runs on the GTX 1650 machine (4GB VRAM, Debian).

Exposes three endpoints:
  GET  /health    — unauthenticated liveness check
  POST /embed     — BGE-M3 cross-lingual text embedding
  POST /classify  — DeBERTa-v3-large-MNLI entailment classification

BGE-M3 is loaded via HuggingFace transformers (AutoModel + AutoTokenizer),
which uses the safetensors format by default. This avoids the torch.load
pickle vulnerability (CVE-2025-32434) that affects FlagEmbedding's loader,
and is fully compatible with PyTorch 2.5.1+.

VRAM budget (GTX 1650, 4GB):
  BGE-M3 (FP16)              ~ 1.1 GB
  DeBERTa-v3-large-MNLI      ~ 0.9 GB
  PyTorch overhead            ~ 0.3 GB
  Total                       ~ 2.3 GB  (1.7 GB headroom)

Run with:
  source venv/bin/activate
  uvicorn main:app --host 0.0.0.0 --port 8001
"""

import time
from contextlib import asynccontextmanager

import structlog
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer, pipeline

log = structlog.get_logger(__name__)

_embed_tokenizer = None
_embed_model = None
_nli_pipeline = None
_device = None


def _average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_tokenizer, _embed_model, _nli_pipeline, _device

    if torch.cuda.is_available():
        _device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("gpu.detected", name=gpu_name, vram_gb=round(vram_gb, 1))
    else:
        _device = "cpu"
        log.warning("gpu.not_found", message="CUDA not available - running on CPU.")

    log.info("bgem3.loading")
    t0 = time.time()
    try:
        _embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        _embed_model = AutoModel.from_pretrained(
            "BAAI/bge-m3",
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        )
        _embed_model.eval()
        _embed_model.to(_device)
        log.info("bgem3.ready", elapsed_s=round(time.time() - t0, 1))
    except Exception as e:
        log.error("bgem3.load_failed", error=str(e))

    log.info("deberta.loading")
    t0 = time.time()
    try:
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-large",
            device=0 if _device == "cuda" else -1,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        )
        log.info("deberta.ready", elapsed_s=round(time.time() - t0, 1))
    except Exception as e:
        log.error("deberta.load_failed", error=str(e))

    if _device == "cuda":
        used_gb = torch.cuda.memory_allocated(0) / 1e9
        reserved_gb = torch.cuda.memory_reserved(0) / 1e9
        log.info("vram.usage", allocated_gb=round(used_gb, 2), reserved_gb=round(reserved_gb, 2))

    log.info("node.ready", message="Secondary node is ready to serve requests.")
    yield

    log.info("node.shutdown")
    del _embed_model, _embed_tokenizer, _nli_pipeline
    if _device == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(
    title="Mnemonic Secondary Node",
    description="BGE-M3 embedding and DeBERTa NLI classification service.",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
)


class EmbedRequest(BaseModel):
    text: str = Field(description="Text to embed. Can be in EN, PT, DE, or ZH.", max_length=8192)
    normalize: bool = Field(default=True, description="L2-normalize for cosine similarity in Qdrant.")


class EmbedResponse(BaseModel):
    vector: list[float]
    model: str
    dimensions: int
    token_count: int
    latency_ms: int


class ClassifyRequest(BaseModel):
    premise: str = Field(description="The source chunk text.", max_length=4096)
    hypothesis: str = Field(description="The generated claim to verify.", max_length=2048)


class ClassifyResponse(BaseModel):
    label: str
    scores: dict[str, float]
    model: str
    latency_ms: int


@app.get("/health", tags=["system"])
async def health() -> dict:
    models_loaded = []
    if _embed_model is not None:
        models_loaded.append("bge-m3")
    if _nli_pipeline is not None:
        models_loaded.append("deberta-v3-large-mnli")

    vram_info = {}
    if _device == "cuda" and torch.cuda.is_available():
        vram_info = {
            "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }

    return {
        "status": "ok" if len(models_loaded) == 2 else "degraded",
        "models_loaded": models_loaded,
        "device": _device,
        "vram": vram_info,
    }


@app.post("/embed", response_model=EmbedResponse, tags=["embedding"])
async def embed(request: EmbedRequest) -> EmbedResponse:
    if _embed_model is None or _embed_tokenizer is None:
        raise HTTPException(status_code=503, detail="BGE-M3 model is not loaded.")

    t0 = time.time()

    encoded = _embed_tokenizer(
        [request.text],
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    ).to(_device)

    token_count = int(encoded["attention_mask"].sum().item())

    with torch.no_grad():
        outputs = _embed_model(**encoded)

    vector = _average_pool(outputs.last_hidden_state, encoded["attention_mask"])

    if request.normalize:
        vector = F.normalize(vector, p=2, dim=1)

    vector_list = vector[0].cpu().float().tolist()
    latency_ms = int((time.time() - t0) * 1000)

    log.info("embed.complete", text_len=len(request.text), token_count=token_count, latency_ms=latency_ms)

    return EmbedResponse(
        vector=vector_list,
        model="bge-m3",
        dimensions=len(vector_list),
        token_count=token_count,
        latency_ms=latency_ms,
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["nli"])
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    if _nli_pipeline is None:
        raise HTTPException(status_code=503, detail="DeBERTa NLI model is not loaded.")

    t0 = time.time()

    result = _nli_pipeline(
        sequences=request.premise,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",
    )

    scores = {
        label.upper(): round(score, 4)
        for label, score in zip(result["labels"], result["scores"])
    }
    top_label = max(scores, key=scores.get)
    latency_ms = int((time.time() - t0) * 1000)

    log.info("classify.complete", label=top_label, confidence=scores[top_label], latency_ms=latency_ms)

    return ClassifyResponse(
        label=top_label,
        scores=scores,
        model="deberta-v3-large-mnli",
        latency_ms=latency_ms,
    )