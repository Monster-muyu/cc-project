# VRAM 显存计算工具

部署 LLM 前估算显存:选模型 + 量化 + 显卡 + 并行/并发参数 → 能不能跑、最多扛几路并发、KV 池容量多少。

仓库: http://192.168.243.204/root/vram-calc · 设计文档: `docs/design.md` · 测试: 23 项

## 环境搭建(Windows + Anaconda)

```powershell
& "E:\ANACONDA\Scripts\conda.exe" create -n vram-calc python=3.11 -y
& "E:\ANACONDA\envs\vram-calc\python.exe" -m pip install -e . --no-build-isolation
& "E:\ANACONDA\envs\vram-calc\python.exe" -m pip install pytest
```

> `--no-build-isolation` 必带:构建隔离的临时 venv 在本机拉不到 setuptools。

## 启动

```powershell
& "E:\ANACONDA\envs\vram-calc\python.exe" -m uvicorn vram_calc.web.app:app --host 127.0.0.1 --port 8000
```
浏览器开 http://127.0.0.1:8000/ (改前端后 Ctrl+Shift+R 硬刷新)。

## 测试与标定

```powershell
& "E:\ANACONDA\envs\vram-calc\python.exe" -m pytest -q
& "E:\ANACONDA\envs\vram-calc\python.exe" scripts\calibrate.py                 # 公开参考对账
& "E:\ANACONDA\envs\vram-calc\python.exe" scripts\calibrate.py --real-kv-tokens 152000  # + 实测
```

## 功能速览
- **计算模型**:vLLM 分页 KV 池(显存利用率扣权重→池容量→max_kv_tokens);
  verdict 三档 = OOM 放不下 / 能跑·会限流 / 放得下
- **模型库**:精选 23+;添加/批量导入支持 **HuggingFace(参数量精确)与 ModelScope(国内直连)**
- **量化**:FP16/FP8/INT4/INT8/GGUF/EXL2;AWQ/GPTQ 预量化仓库自动识别并锁定
- **并行**:显卡数量自由输入(dense 自动 TP,MoE 自动 EP),高级手动 TP/PP/EP 到 64
- **推荐**:并发↔上下文对照表 + 保并发/保上下文/保守推荐(照着输保证放得下)

## 已知简化(升级路径见代码内 ponytail 注释)
- 激活系数、引擎开销画像为经验值 → `scripts/calibrate.py` 对账后调整
- DeepSeek MLA 的 KV 为近似;VL 模型忽略视觉编码器(轻微低估)
- 异构多卡、训练/LoRA、tokens/s 不做(v1 范围外)
- vLLM 部分版本在 max_model_len > 单序列池容量时会拒绝启动(工具显示"能跑·会限流"指运行时不 OOM)
