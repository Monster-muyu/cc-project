"""Entity stores + HuggingFace model fetch.

- Curated models/gpus ship as bundled JSON arrays (read-only).
- User-added entities live as one JSON file each under ~/.vram_calc/.
- Runtime list = bundled ∪ user, user overrides bundled by id.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.estimator import ModelSpec, GpuSpec
from ..core.arch_resolver import resolve_arch, detect_quant

PKG_DIR = Path(__file__).resolve().parent.parent          # .../vram_calc/
BUNDLED_DIR = PKG_DIR / "data"
USER_DIR = Path.home() / ".vram_calc"

STANDARD_QUANTS = ("fp16", "bf16", "fp8", "int8", "int4",
                   "gguf-q4_k_m", "gguf-q5_k_m", "gguf-q8_0", "exl2")


def infer_quant_from_id(model_id: str) -> str:
    """Infer a fixed quant from a repo id/name — fallback for entries saved
    before quant auto-detection, or GGUF repos (quant only in filename)."""
    s = model_id.lower()
    for tag, q in (("q8_0", "gguf-q8_0"), ("q6_k", "gguf-q6_k"), ("q5_k_m", "gguf-q5_k_m"),
                   ("q4_k_m", "gguf-q4_k_m"), ("q3_k_m", "gguf-q3_k_m"), ("q2_k", "gguf-q2_k")):
        if tag in s:
            return q
    if "exl2" in s:
        return "exl2"
    if "awq" in s or "gptq" in s:
        return "int8" if ("int8" in s or "8bit" in s) else "int4"
    if "int8" in s or "8bit" in s:
        return "int8"
    if "int4" in s or "4bit" in s:
        return "int4"
    if "fp8" in s:
        return "fp8"
    return ""


class EntityStore:
    """Bundled JSON array + user dir of per-entity JSON files, merged (user wins)."""

    def __init__(self, bundled_name: str, user_subdir: str):
        self.bundled_path = BUNDLED_DIR / bundled_name
        self.user_dir = USER_DIR / user_subdir

    def _load_bundled(self) -> dict[str, dict]:
        if not self.bundled_path.exists():
            return {}
        data = json.loads(self.bundled_path.read_text(encoding="utf-8"))
        return {e["id"]: e for e in data}

    def _load_user(self) -> dict[str, dict]:
        if not self.user_dir.exists():
            return {}
        out: dict[str, dict] = {}
        for f in self.user_dir.glob("*.json"):
            try:
                e = json.loads(f.read_text(encoding="utf-8"))
                out[e["id"]] = e
            except (json.JSONDecodeError, KeyError):
                continue   # ponytail: skip malformed user file instead of crashing
        return out

    def list(self) -> list[dict]:
        return list({**self._load_bundled(), **self._load_user()}.values())

    def get(self, entity_id: str) -> dict | None:
        return self._load_user().get(entity_id) or self._load_bundled().get(entity_id)

    def save(self, entity: dict) -> Path:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        path = self.user_dir / f"{_safe(entity['id'])}.json"
        path.write_text(json.dumps(entity, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


models = EntityStore("models.json", "models")
gpus = EntityStore("gpus.json", "gpus")


# ---- models ----
def list_models() -> list[ModelSpec]:
    return [_dict_to_model(e) for e in models.list()]


def get_model(mid: str) -> ModelSpec | None:
    e = models.get(mid)
    return _dict_to_model(e) if e else None


def save_model(m: ModelSpec) -> Path:
    return models.save(_model_to_dict(m))


# ---- gpus ----
def list_gpus() -> list[GpuSpec]:
    return [_dict_to_gpu(e) for e in gpus.list()]


def get_gpu(gid: str) -> GpuSpec | None:
    e = gpus.get(gid)
    return _dict_to_gpu(e) if e else None


def save_gpu(g: GpuSpec) -> Path:
    return gpus.save(_gpu_to_dict(g))


# ---- conversions ----
def _dict_to_model(e: dict) -> ModelSpec:
    return ModelSpec(
        id=e["id"], name=e.get("name", e["id"]),
        params_b=e["params_b"], layers=e["layers"], hidden_dim=e["hidden_dim"],
        attn_heads=e["attn_heads"], kv_heads=e["kv_heads"], head_dim=e["head_dim"],
        vocab_size=e.get("vocab_size", 0), num_experts=e.get("num_experts", 0),
        expert_params_b=e.get("expert_params_b", 0.0),
        quantizations=tuple(e.get("quantizations", STANDARD_QUANTS)),
        category=e.get("category", "llm"),
        quant=e.get("quant", ""),
    )


def _model_to_dict(m: ModelSpec) -> dict:
    return {"id": m.id, "name": m.name, "params_b": m.params_b, "layers": m.layers,
            "hidden_dim": m.hidden_dim, "attn_heads": m.attn_heads, "kv_heads": m.kv_heads,
            "head_dim": m.head_dim, "vocab_size": m.vocab_size,
            "num_experts": m.num_experts, "expert_params_b": m.expert_params_b,
            "quantizations": list(STANDARD_QUANTS), "category": m.category, "quant": m.quant}


def _dict_to_gpu(e: dict) -> GpuSpec:
    return GpuSpec(id=e["id"], name=e.get("name", e["id"]), vram_gb=e["vram_gb"],
                   memory_bw_gbps=e.get("memory_bw_gbps", 0.0),
                   fp16_tflops=e.get("fp16_tflops", 0.0),
                   supports_fp8=e.get("supports_fp8", False),
                   supports_bf16=e.get("supports_bf16", True))


def _gpu_to_dict(g: GpuSpec) -> dict:
    return {"id": g.id, "name": g.name, "vram_gb": g.vram_gb,
            "memory_bw_gbps": g.memory_bw_gbps, "fp16_tflops": g.fp16_tflops,
            "supports_fp8": g.supports_fp8, "supports_bf16": g.supports_bf16}


# ---- HuggingFace fetch (needs network) ----
def fetch_model_preview(repo_id: str, category: str = "llm") -> ModelSpec:
    """Fetch param count (safetensors) + arch (config.json) from HF.

    Returns a ModelSpec for the user to preview/edit before saving.
    Zero weight download: only the small model_info API call + config.json.
    """
    from huggingface_hub import model_info, hf_hub_download

    info = model_info(repo_id)
    params_b = _params_from_info(info)
    cfg_path = hf_hub_download(repo_id, "config.json")
    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    arch = resolve_arch(config)

    if not params_b:                       # safetensors params unavailable -> estimate
        params_b = _estimate_params_from_arch(arch, config)

    return ModelSpec(
        id=repo_id, name=repo_id.split("/")[-1],
        params_b=round(params_b, 3), quantizations=STANDARD_QUANTS,
        category=category, quant=detect_quant(config), **arch,
    )


def fetch_and_save_many(repo_ids: list[str], category: str = "llm") -> dict:
    """Bulk: fetch + save each repo. Returns {"saved": [...], "failed": [...]}."""
    saved, failed = [], []
    for rid in repo_ids:
        rid = rid.strip()
        if not rid:
            continue
        try:
            save_model(fetch_model_preview(rid, category=category))
            saved.append(rid)
        except Exception as e:   # one bad repo must not abort the batch
            failed.append({"id": rid, "error": f"{type(e).__name__}: {e}"})
    return {"saved": saved, "failed": failed}


def _params_from_info(info) -> float | None:
    """Pull total parameter count from HF safetensors metadata (per-dtype)."""
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    if isinstance(st, dict):                       # {"parameters": {dtype: count}, ...}
        p = st.get("parameters")
        if isinstance(p, dict) and p:
            return sum(p.values()) / 1e9
    if isinstance(st, list):                       # per-file metadata objects
        total = 0
        for s in st:
            pc = getattr(s, "parameter_count", None)
            if isinstance(pc, dict):
                total += sum(pc.values())
            elif isinstance(pc, int):
                total += pc
        if total:
            return total / 1e9
    return None   # never derive from `total` bytes -- breaks on mixed/quantized dtypes


def _estimate_params_from_arch(arch: dict, config: dict) -> float:
    """Rough param-count fallback when safetensors metadata is absent."""
    h = arch["hidden_dim"] or 0
    L = arch["layers"] or 0
    V = arch["vocab_size"] or 0
    inter = config.get("intermediate_size") or 4 * h
    dense = V * h + L * (4 * h * h + 3 * h * inter)
    expert = arch["num_experts"] * 3 * h * inter * L
    return (dense + expert) / 1e9
