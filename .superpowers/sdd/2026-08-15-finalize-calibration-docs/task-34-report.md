# Task 3+4 Report

## Task 3: 标定对账脚本

**Status:** Done

### calibrate.py output

```
用例                                               预测           参考       误差  判定
------------------------------------------------------------------------------
Llama3-8B fp16 权重                              16.1         16.1     0.0%  ✅ (GB, 容差0.1%)
Llama3-8B fp16 @4090 固定占用                      19.9         20.0     0.7%  ✅ (GB, 容差10.0%)
Mixtral 8x7B fp16 权重                           93.4         93.4     0.0%  ✅ (GB, 容差0.5%)
DeepSeek-V3 fp8 开销(封顶)                          7.5          7.5     0.0%  ✅ (GB, 容差15.0%)
------------------------------------------------------------------------------
全部达标 ✅
```

- Exit code: 0
- Pass: 4/4, Fail: 0/4
- Llama3-8B fixed occupancy: predicted 19.9 vs ref 20.0, error 0.7% (well within 10%)
- All arithmetic cases exact match (0.0% error)

### Commit

```
feat:标定对账脚本(公开参考+用户实测KV池入口,超容差退出码1)
```

## Task 4: README

**Status:** Done

### Commit

```
docs:README(环境搭建/启动/测试标定/功能/已知简化)
```

## Concerns

None. All 4 calibration cases pass cleanly with zero tuning needed.
