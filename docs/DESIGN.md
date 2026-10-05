# mini-rl Orch 设计

## 1. 边界

`orch` 是一个只支持 disaggregated GRPO 的最小原型。它只从 Meshy 借用两点：

- Ray-free 的显式 service topology；
- 角色驱动的常驻循环，不由中央 driver 执行训练主循环。

Staleness 语义全部来自 StaleFlow/PSRL。运行时没有 Meshy staleness controller，
也没有其他同步或训练模式入口。

## 2. Meshy 形状

```text
Recipe -> ServiceGroup -> Topology -> Ignitor
                                      |
                                      v
                            Service -> Worker -> Engine
```

| 层 | 职责 |
| --- | --- |
| `Recipe` | 声明角色、副本、依赖和资源 |
| `Topology` | 由 recipe 与全局 GPU 清单纯函数地确定实例、GPU 和 endpoint |
| `Ignitor` | 每卡一个，只启动归属本卡的角色子进程并守护生命周期 |
| `Service` | 一个角色及其资源边界 |
| `Worker` | 角色自己的常驻循环 |
| `Engine` | 推理或训练后端接口 |

launcher 在每台机器运行同一条命令，通过一个多机 `torchrun` 在每张卡上启动
ignitor。所有 ignitor all-gather 全局 GPU 清单后独立计算相同 topology，不依赖 Ray、
注册中心或中心任务派发。`Ignitor` 启动完成后不参与 Reserve、生成、训练或参数同步。

## 3. StaleFlow 角色

```text
                                  status snapshot
                           +----------------------------+
                           |                            v
RolloutWorker -- Reserve / Occupy --> PSManager   RolloutCoordinator
      |                              |                   |
      |                              |              SYNC / ABORT
      |                              v                   |
      |                      StalenessInventory          |
      |                                                  |
      +-- trajectory --> TrajectoryServer --> Trainer <--+
      |                                          |
      +-- pull weights --> ParameterServer <-----+ push weights
```

### ParameterServerService

该服务包含两个不同职责：

- `ParameterServer`：保存按版本索引的模型 handle；
- `PSManager`：通过 `StalenessInventory` 管理 Reserve、Occupy、Lease、Consume 元数据。

参数必须先成功 push，PSManager 才允许 consume 对应训练 buffer。因此对 rollout
可见的 committed version 不会指向缺失的参数版本。

### TrajectoryServerService

保存 tokens、logprobs、reward、advantage 等 payload。数据按
`<partition>.buffer.<id>` 隔离，并通过 reservation membership 校验 trainer 取到的
batch。PSManager 只保存 reservation id，不转发 trajectory 字节。

### RolloutCoordinatorService

保存每个 producer 最新的状态快照和有序 command mailbox。旧 `snapshot_seq` 会被
丢弃；producer epoch 用于 fencing 重启前的实例。命令只保证 per-producer 顺序。

它是逻辑单例控制面服务，但不是执行 orchestrator：它不会调用 rollout/trainer
主循环，也不承载 trajectory 或参数。

当前命令只有：

- `SYNC(target_version)`：producer 在安全边界取消尚未发布的 reservation，从 PS 拉取
  指定版本，然后用 command sequence 回执；
- `ABORT(group_ids)`：取消并 fence 指定的在途 rollout group。

没有 `SLEEP/WAKE` 和模型 offload。

## 4. Reserve / Occupy / Consume

设 rollout group 使用的 behavior version 为 `v`，staleness 上限为 `s`，最终进入的
训练 buffer 为 `b`。StalenessInventory 始终维持：

```text
v <= b <= v + s
```

协议分三步：

1. **Reserve latest**：生成前预占最晚可行 buffer，避免无限制产生旧 trajectory。
2. **Occupy earliest**：整组 GRPO 数据完成后移动到最早可行 buffer，优先填满 frontier。
3. **Consume frontier**：trainer 只能 lease frontier；训练后依次 push 参数、consume
   metadata、清理 trajectory payload。

Reservation 带 `owner_epoch`。一个 group abort 后重新分配时，旧实例迟到的 Occupy
会被 fencing 拒绝。

## 5. 一次完整数据流

```text
1. rollout -> coordinator : ProducerSnapshot(v, running, waiting, seq)
2. rollout -> PSManager   : Reserve(group, v)
3. rollout -> inference   : generate complete GRPO group
4. rollout -> PSManager   : Occupy(reservation)
5. rollout -> trajectory  : PutGroup(payload, training_buffer)
6. trainer -> PSManager   : Lease(frontier)
7. trainer -> trajectory  : Fetch(lease membership)
8. trainer -> engine      : Train(batch)
9. trainer -> PS          : Push(v + 1)
10. trainer -> PSManager  : Consume(lease, v + 1)
11. coordinator -> rollout: SYNC(v + 1)
12. rollout -> PS         : Pull(v + 1)
```

Rollout 和 trainer 可并发推进，中央没有逐 step 驱动逻辑。

## 6. 当前限制

- 单 trainer 和单 logical PSManager；
- 每个 rollout replica 绑定一个独立 inference replica；
- 固定 group-count batch；
- 默认 recipe 使用 toy inference/training engine；16 卡 smoke recipe 会真实加载并常驻
  完整 40 层模型，但 forward/backward 和参数更新仍为 fake；
- JSON/HTTP 是当前最小跨节点 transport，不适合传输生产规模 tensor；
- 不包含 lease timeout、复制和故障恢复；
- 不包含 offload、colocation 或同步训练模式。

生产化时可以替换 engine、ParameterServer、TrajectoryServer 和 RPC transport，而不改变
Service/Worker 角色循环及 StaleFlow 协议边界。
