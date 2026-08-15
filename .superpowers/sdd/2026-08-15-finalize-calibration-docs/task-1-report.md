# Task 1 Report: 设计文档同步到 v2.0

## Status: DONE

## Changes Applied (8/8 steps)
1. **Header** — v1.1→v2.0, full changelog block
2. **§5.2** — Static KV formula → vLLM paged pool model (KV预算/max_kv_tokens)
3. **§5.3** — activation driven by max_num_batched_tokens, not context×concurrency
4. **§5.4** — appended cap_gb封顶说明
5. **§5.8** — new section: verdict true semantics (over/ok/tight with KV pool logic)
6. **§6.1** — dual-source fetch (HF + ModelScope), auto quant lock, VL/MoE notes
7. **§7.1+§7.2** — GPU count free input, gpu_memory_utilization slider, sensitivity chart "KV demand vs budget"
8. **§8** — /api/calc response (kv_budget_gb, max_kv_tokens, fixed-only total_gb); /api/sweep (kv_budget_gb, KV demand semantics)

## Coherence Check
- §5.2 paged pool model consistent with §5.8 verdict semantics
- §8 API responses match new breakdown field names (kv_cache not kv, per_gpu_gb not per_gpu)
- §7 UI controls align with §8 request params (gpu_memory_utilization, max_num_batched_tokens)
- §6.1 dual-source matches §3 tech stack (huggingface_hub still listed, ModelScope is addition)
- No contradictions with remaining v1 sections (§5.1 weights, §5.5 parallel, §5.6 offload unchanged and still correct)

## Commit
`docs:设计文档同步v2.0(KV池模型/verdict真实语义/双源/UI现状)` — 1 file, +61 -24

## Concerns
- §2 "v1 纳入" and §11 "已知简化" still reference old KV×并发最坏假设; these are legacy scope notes, not computational, so not updated per brief scope.
- §8 `/api/calc` request params still list old field set (no gpu_memory_utilization or max_num_batched_tokens); brief did not cover request params update.
