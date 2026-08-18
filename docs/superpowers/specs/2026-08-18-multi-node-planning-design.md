# 多机部署规划 — 设计文档

日期：2026-08-18
状态：已与用户逐节确认（6 节全过）

## 1. 背景与目标

现有工具只支持单机多卡估算。本设计扩展为多机部署规划：用户给出服务器清单（每台机器
有哪些卡），工具自动规划跨机并行切分，给出每机显存账本、候选方案对比和可直接执行的
vLLM 启动命令。

## 2. 已确认的决策

| 决策点 | 结论 |
|---|---|
| 异构范围 | 数据模型按最大通用性设计（每机 GPU 槽位清单，机内混插可表达）；规划器 v1 支持机间异构，机内混插只警告不规划 |
| 并行策略 | 自动规划为主（带推理过程），手动微调后期再加 |
| 输出形态 | 估算 + 可执行启动命令（ray + vllm serve） |
| 页面关系 | 独立新页签「多机规划」，与单机页共用估算引擎与样式 |
| 方案数量 | top 3 个候选方案对比（最少机器 / 吞吐优先 / 跨机大实例），不足 3 时按实际数 |
| 实现架构 | 后端规划器模块 `core/planner.py`，前端只做输入和渲染（否掉前端编排与约束求解器） |

## 3. 数据模型

```python
# core/cluster.py（或并入 estimator.py）
@dataclass(frozen=True)
class GpuCount:
    gpu_id: str
    count: int

@dataclass(frozen=True)
class ServerSpec:
    id: str
    name: str
    host: str = ""                  # 仅用于生成启动命令，可空
    gpus: tuple[GpuCount, ...] = () # 列表结构：混插天然可表达
```

- 存储：复用 `EntityStore`，用户服务器存 `~/.vram_calc/servers/*.json`，
  bundled 种子为空数组。与 models/gpus 同一套 CRUD 机制。
- 规划请求 `PlanInput`：model_id、server_ids、context_len、concurrency、
  quant、kv_quant、gpu_util。引擎 v1 固定 vllm。
- 规划结果 `Plan`：name、badges、topo(tp/pp/ep/dp)、assignment（每机→角色+卡数）、
  ledger（每机 breakdown，复用现有 EstimateResult 结构）、verdict、why、commands。

## 4. 规划器规则（core/planner.py）

### 候选枚举（按序生成，全部过约束过滤）

1. **单机方案**：每台机器独立（TP=该机卡数，PP=1）→「最少机器」候选
2. **DP 副本方案**：机器分组，每组组成一个完整实例，多组=多副本 →「吞吐优先」
3. **跨机 PP 方案**：TP 尽量机内、PP 跨机拼接（如 2 机 × TP4 → 8 卡单实例）
4. **跨机 TP 方案**：仅当上述全放不下时生成，标黄「跨机 TP 慢」

### 硬约束（违反淘汰）

- TP 组内 GPU 型号一致；TP 整除注意力头数（复用现有校验）
- PP 各 stage 卡数 = TP 卡数（vLLM stage 内一致要求）
- MoE 模型优先 EP 变体（专家跨卡均摊）
- FP8 KV + 不支持 FP8 的卡 → 候选降级警告（复用现有 FP8 门控）

### 评分排序（每候选举调 estimate() 拿真实数字）

verdict 等级 > 机器利用率（少开机优先）> KV 余量 > 跨机惩罚。
输出 top 3（去重），每方案附规则命中记录生成的 why 推理文本。

## 5. API

```
GET  /plan                  → 多机规划页（HTML）
GET  /api/servers           → 服务器列表
POST /api/servers           → 保存/更新
DELETE /api/servers/{id}    → 删除
POST /api/plan              → 规划（纯内存，几十次 estimate() 毫秒级，不做异步）
```

`/api/plan` 请求：`{model_id, quant, kv_quant, context_len, concurrency,
gpu_util, server_ids}`。响应：top 3 Plan 数组。
页签导航灰色「多机规划」激活为真链接。

## 6. 页面（templates/plan.html + static/plan.js）

左栏：服务器清单卡（条目+GPU 徽标+混插红警告+勾选参与；添加服务器弹窗：名称/host/
多行 GPU 型号+卡数）＋规划目标卡（模型/上下文(k,m)/并发/量化/KV 量化/利用率滑杆/
开始规划按钮）。

右栏：空态提示 → top 3 方案卡（徽标/拓扑行/why/每机账本表+占用条/折叠命令块带复制）。
参数变化 debounce 800ms 自动重算。

公共函数（parseContext、下拉加载、fmtGB 等）从 app.js 提取 common.js，两页共用。
样式进 style.css 全局段（.plan 系列类），页面无内联 style。

## 7. 启动命令生成（core/commands.py）

输入 Plan，输出命令块列表（标题+代码文本），纯字符串拼装：

- 单机：一块 vllm serve（参数值全部来自规划请求）
- DP 多副本：按副本分块，相同命令合并+网关注释
- 跨机 PP/TP：ray head → ray worker（host 空则 `<server-id>-IP` 占位+注释）→
  vllm serve --distributed-executor-backend ray；块尾附环境前提注释
- MoE EP 方案追加 --enable-expert-parallel

## 8. 测试

- pytest 单测：规划器（同构单机/必须跨机/DP 分组/混插警告/头数整除淘汰/top3 排序，
  小型 fixture 直接断言数字）；命令生成文本快照；服务器 EntityStore 存取
- API：TestClient 走 /api/plan
- 页面：Playwright 手动验证一轮，不写自动化 UI 测试（与单机页策略一致）

## 9. 不做（YAGNI）

- 手动微调切分界面（后期）
- 机内混插参与规划（数据可存，规划器只警告）
- 多引擎（sglang/llama.cpp 多机路径各异，v1 只 vllm）
- 网络带宽建模（千兆/万兆/RDMA 影响性能不影响显存，页面文字提示即可）
- 规划结果持久化（会话内使用，无收藏/历史）
