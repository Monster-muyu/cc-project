# SDD ledger — plan: docs/superpowers/plans/2026-08-15-finalize-calibration-docs.md

BASE: 1f8031c

## Pre-flight conflict scan

| 检查项 | 结论 |
|---|---|
| Task1(文档) vs Task2(engines.py) | Task1 §5.4 提前写了"封顶 cap_gb"字样,依赖 Task2 先落地才能文档为真。**Ruling: 合并进同一次派发顺序——Task2 先执行,Task1 文档后写,交换计划内的任务顺序(Task1↔Task2)。理由:文档描述的代码必须已存在;代价:无,两者无文件交集** |
| Task2 测试(23 passed) vs Task5 全量(23) | Task3/4 不加测试,数字一致 ✓ |
| Task3 calibrate.py 引用 EngineProfile cap 效果 | 依赖 Task2 的封顶(开销用例 7.5 参考值);Task2 先行即可 ✓ |
| Task4 README "测试: 23 项" | 与 Task2 后的计数一致 ✓ |
| Task5 push 到 origin/main | 计划明示要求,属外部副作用但已获用户持续授权(整轮迭代都在 push main)。**Ruling: 执行** |
| worktree | 未建独立 worktree。**Ruling: 在 main 上直接执行——本项目整个开发周期都在 main 上迭代并推送,GitLab 单分支流;建 worktree 反而脱离用户使用习惯。代价:若出错可 git revert** |

Tasks: 1文档同步 2开销封顶 3标定脚本 4README 5收尾验证
执行顺序(见 Ruling): 2 → 1 → 3 → 4 → 5

Task 2: complete (commits 1f8031c..e1df574, review clean)
Task 1: complete (commits e1df574..e3f34a7, review clean with follow-ups)
Task 1: minor (deferred): §5.5 静态每卡KV公式与§5.2池模型矛盾; §5.7 旧verdict缺指向§5.8的桥接注; §8 请求参数未含 gpu_memory_utilization/max_num_batched_tokens; §2 残留旧口径
