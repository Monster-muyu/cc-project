"""Entity stores + HuggingFace model fetch.

- Curated models/gpus ship as bundled JSON arrays (read-only).
- User-added entities live as one JSON file each under ~/.vram_calc/.
- Runtime list = bundled ∪ user, user overrides bundled by id.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..core.estimator import ModelSpec, GpuSpec
from ..core.cluster import ServerSpec, GpuCount
from ..core.arch_resolver import resolve_arch, detect_quant
from ..core.quant import QUANT_BYTES

PKG_DIR = Path(__file__).resolve().parent.parent          # .../vram_calc/
BUNDLED_DIR = PKG_DIR / "data"
USER_DIR = Path(os.environ.get("VRAM_CALC_HOME", str(Path.home() / ".vram_calc")))


# ---- calibration (real-log engine-overhead overrides) ----
def load_calibration() -> dict:
    """{"vllm:rtx-3090": {overhead_gb, weights_gib_at, kv_gib_at, util, date}}"""
    p = USER_DIR / "calibration.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_calibration_entry(engine: str, gpu_id: str, entry: dict) -> dict:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    data = load_calibration()
    data[f"{engine}:{gpu_id}"] = entry
    (USER_DIR / "calibration.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

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

    def delete(self, entity_id: str) -> bool:
        p = self.user_dir / f"{_safe(entity_id)}.json"
        if p.exists():
            p.unlink()
            return True
        return False


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


# ---- servers ----
servers = EntityStore("servers.json", "servers")


def list_servers() -> list[ServerSpec]:
    return [_dict_to_server(e) for e in servers.list()]


def get_server(sid: str) -> ServerSpec | None:
    e = servers.get(sid)
    return _dict_to_server(e) if e else None


def save_server(s: ServerSpec) -> Path:
    return servers.save({"id": s.id, "name": s.name, "host": s.host,
                         "gpus": [{"gpu_id": g.gpu_id, "count": g.count} for g in s.gpus]})


def delete_server(sid: str) -> bool:
    return servers.delete(sid)


def _dict_to_server(e: dict) -> ServerSpec:
    return ServerSpec(id=e["id"], name=e.get("name", e["id"]),
                      host=e.get("host", ""),
                      gpus=tuple(GpuCount(**g) for g in e.get("gpus", [])))


# ---- conversions ----
def _dict_to_model(e: dict) -> ModelSpec:
    return ModelSpec(
        id=e["id"], name=e.get("name", e["id"]),
        params_b=e["params_b"], layers=e["layers"], hidden_dim=e["hidden_dim"],
        attn_heads=e["attn_heads"], kv_heads=e["kv_heads"], head_dim=e["head_dim"],
        kv_layers=e.get("kv_layers", 0), linear_heads=e.get("linear_heads", 0),
        vocab_size=e.get("vocab_size", 0), num_experts=e.get("num_experts", 0),
        expert_params_b=e.get("expert_params_b", 0.0),
        quantizations=tuple(e.get("quantizations", STANDARD_QUANTS)),
        category=e.get("category", "llm"),
        quant=e.get("quant", ""),
    )


def _model_to_dict(m: ModelSpec) -> dict:
    return {"id": m.id, "name": m.name, "params_b": m.params_b, "layers": m.layers,
            "hidden_dim": m.hidden_dim, "attn_heads": m.attn_heads, "kv_heads": m.kv_heads,
            "head_dim": m.head_dim, "kv_layers": m.kv_layers,
            "linear_heads": m.linear_heads, "vocab_size": m.vocab_size,
            "num_experts": m.num_experts, "expert_params_b": m.expert_params_b,
            "quantizations": list(STANDARD_QUANTS), "category": m.category, "quant": m.quant}


def _dict_to_gpu(e: dict) -> GpuSpec:
    return GpuSpec(id=e["id"], name=e.get("name", e["id"]), vram_gb=e["vram_gb"],
                   memory_bw_gbps=e.get("memory_bw_gbps", 0.0),
                   fp16_tflops=e.get("fp16_tflops", 0.0),
                   architecture=e.get("architecture", ""),
                   vendor=e.get("vendor", "nvidia"),
                   supports_fp8=e.get("supports_fp8", False),
                   supports_bf16=e.get("supports_bf16", True))


def _gpu_to_dict(g: GpuSpec) -> dict:
    return {"id": g.id, "name": g.name, "vram_gb": g.vram_gb,
            "memory_bw_gbps": g.memory_bw_gbps, "fp16_tflops": g.fp16_tflops,
            "architecture": g.architecture, "vendor": g.vendor,
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
    try:
        cfg_path = hf_hub_download(repo_id, "config.json")
    except Exception:                     # GGUF 仓库无 config.json -> 文件名反推
        variants = _parse_gguf_variants([
            {"Path": s.rfilename, "Size": getattr(s, "size", None) or 0}
            for s in model_info(repo_id, files_metadata=True).siblings or []])
        if not variants:
            raise

        def _base_cfg(base: str) -> str:
            return Path(hf_hub_download(base, "config.json")).read_text(encoding="utf-8")

        return _gguf_model_spec(repo_id, repo_id, variants, category, _base_cfg)
    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    arch = resolve_arch(config)

    if not params_b:                       # safetensors params unavailable -> estimate
        params_b = _estimate_params_from_arch(arch, config)

    return ModelSpec(
        id=repo_id, name=repo_id.split("/")[-1],
        params_b=round(params_b, 3), quantizations=STANDARD_QUANTS,
        category=category, quant=detect_quant(config), **arch,
    )


def fetch_and_save_many(repo_ids: list[str], category: str = "llm", source: str = "hf") -> dict:
    """Bulk: fetch + save each repo. Returns {"saved": [...], "failed": [...]}."""
    saved, failed = [], []
    for rid in repo_ids:
        rid = rid.strip()
        if not rid:
            continue
        try:
            m = fetch_modelscope(rid, category=category) if source == "ms" \
                else fetch_model_preview(rid, category=category)
            save_model(m)
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


# ---- ModelScope source (no modelscope SDK needed -- plain REST) ----
_MS_BASE = "https://modelscope.cn/api/v1"


# ---- GGUF repos: no config.json/safetensors -- params & quants from filenames ----
# GGUF 仓库只有 .gguf 文件（多套量化变体混放），参数量从最可信变体的字节数反推。
_GGUF_BUCKETS = (
    ("gguf-q8_0", ("Q8",)), ("gguf-q6_k", ("Q6",)), ("gguf-q5_k_m", ("Q5",)),
    ("gguf-q4_k_m", ("Q4", "IQ4")), ("gguf-q3_k_m", ("Q3", "IQ3")),
    ("gguf-q2_k", ("Q2", "IQ2")), ("bf16", ("BF16",)), ("fp16", ("F16",)),
)


def _gguf_bucket(tag: str) -> str | None:
    for key, prefixes in _GGUF_BUCKETS:
        if tag.upper().startswith(prefixes):
            return key
    return None                      # F32 等稀有格式跳过


def _parse_gguf_variants(files: list[dict]) -> dict[str, int]:
    """文件列表 -> {量化桶: 总字节}。mmproj(多模态投影)/imatrix 排除，分片合并。"""
    out: dict[str, int] = {}
    for f in files:
        path = f.get("Path") or f.get("Name") or ""
        if not path.endswith(".gguf"):
            continue
        base = path.rsplit("/", 1)[-1]
        if base.startswith(("mmproj-", "imatrix")):
            continue
        stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", base)   # 分片后缀合并
        m = re.search(r"-(?:UD-)?([A-Z0-9_]+)\.gguf$", stem)      # UD- = Unsloth Dynamic
        if not m:
            continue
        key = _gguf_bucket(m.group(1))
        if key:
            out[key] = out.get(key, 0) + int(f.get("Size") or 0)
    return out


def _params_from_variants(variants: dict[str, int]) -> float:
    """优先 BF16/F16/Q8_0/Q6_K 反推（bpw 已知最准），兜底任一桶。"""
    for key in ("bf16", "fp16", "gguf-q8_0", "gguf-q6_k"):
        if variants.get(key):
            return variants[key] / QUANT_BYTES[key] / 1e9
    for key in sorted(variants):
        if variants[key]:
            return variants[key] / QUANT_BYTES[key] / 1e9
    return 0.0


def _typical_arch(params_b: float) -> dict:
    """GGUF 仓库拿不到 config 时的按参数量级兜底架构（预览弹窗可手改）。"""
    if params_b < 6:
        layers, hidden, attn = 28, 3584, 28
    elif params_b < 12:
        layers, hidden, attn = 32, 4096, 32
    elif params_b < 24:
        layers, hidden, attn = 40, 5120, 40
    elif params_b < 45:
        layers, hidden, attn = 64, 5120, 40
    else:
        layers, hidden, attn = 80, 8192, 64
    return {"layers": layers, "hidden_dim": hidden, "attn_heads": attn,
            "kv_heads": 8, "head_dim": 128, "vocab_size": 0,
            "num_experts": 0, "expert_params_b": 0.0, "kv_layers": 0}


def _gguf_model_spec(repo_id: str, spec_id: str, variants: dict[str, int],
                     category: str, fetch_base_config) -> ModelSpec:
    """GGUF 仓库通用落库：参数量从变体字节反推，架构优先取 base 仓库 config。"""
    params_b = _params_from_variants(variants)
    arch = None
    base = re.sub(r"-gguf$", "", repo_id, flags=re.I)
    if base != repo_id:
        try:
            arch = resolve_arch(json.loads(fetch_base_config(base)))
        except Exception:            # base 仓库不存在/无 config -> 按量级兜底
            arch = None
    if not arch or not arch.get("layers"):
        arch = _typical_arch(params_b)
    pref = "gguf-q4_k_m" if "gguf-q4_k_m" in variants else sorted(variants)[0]
    return ModelSpec(
        id=spec_id, name=repo_id.split("/")[-1],
        params_b=round(params_b, 3), quantizations=tuple(sorted(variants)),
        category=category, quant=pref, **arch,
    )


def _ms_get(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url, timeout=20) as resp:   # noqa: S310 (https, fixed host)
        return resp.read()


def fetch_modelscope(repo_id: str, category: str = "llm") -> ModelSpec:
    """Fetch arch from ModelScope config.json; params estimated (~3% error, editable).

    ModelScope's model-info API has no per-dtype parameter breakdown (only
    StorageSize, unreliable for quantized repos), so we estimate from arch.
    GGUF repos have no config.json (404) -> params from .gguf file sizes.
    """
    import urllib.error

    try:
        config = json.loads(_ms_get(f"{_MS_BASE}/models/{repo_id}/repo?FilePath=config.json"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        files = json.loads(_ms_get(
            f"{_MS_BASE}/models/{repo_id}/repo/files?Recursive=true"))["Data"]["Files"]
        variants = _parse_gguf_variants(files)
        if not variants:
            raise ValueError(f"仓库无 config.json 且未找到 .gguf 权重: {repo_id}")
        return _gguf_model_spec(
            repo_id, f"ms/{repo_id}", variants, category,
            lambda base: _ms_get(f"{_MS_BASE}/models/{base}/repo?FilePath=config.json"))

    arch = resolve_arch(config)
    params_b = _estimate_params_from_arch(arch, config)
    return ModelSpec(
        id=f"ms/{repo_id}", name=repo_id.split("/")[-1],
        params_b=round(params_b, 3), quantizations=STANDARD_QUANTS,
        category=category, quant=detect_quant(config), **arch,
    )


def _estimate_params_from_arch(arch: dict, config: dict) -> float:
    """Rough param-count fallback when safetensors metadata is absent."""
    h = arch["hidden_dim"] or 0
    L = arch["layers"] or 0
    V = arch["vocab_size"] or 0
    inter = config.get("intermediate_size") or 4 * h
    dense = V * h + L * (4 * h * h + 3 * h * inter)
    expert = arch["num_experts"] * 3 * h * inter * L
    return (dense + expert) / 1e9
