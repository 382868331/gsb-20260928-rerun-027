# 自动驾驶场景回放系统（第一版）

确定性的场景时间引擎：车辆、行人、信号灯按同一个整数毫秒时钟推进，
支持暂停、跳回过去继续、事件固定顺序与一次性触发、场景版本/种子保存。
不是动画演示——所有状态来自可重放的离散时间步进。

## 核心语义

- 统一时钟：固定步长（默认 100ms），每 tick 先按固定顺序推进
  参与者（车辆在前、行人在后），再按 (tick, seq, event_id) 全序触发事件。
- 暂停/中断：pause() 暂停人工控制；{"type": "pause"} 事件表示
  场景中断，解除后才能继续。
- 定位到过去：seek(t) 从初始场景静默重放到目标时刻重建状态；
  已处理事件的副作用会重放，但事件不会重复登记/重复触发。
- 逐步与跳转一致：advance(n)、run_to(t)、seek(t) 走同一套
  状态转移函数，同一时刻位置、信号、事件日志完全相同。
- 版本与种子：Scenario.version / seed 随 save_run 持久化为 JSON。
- 任意时刻分支：create_branch(parent, tick, BranchChanges(...), name)
  在分支 tick 先按原状态完成移动，再固定顺序应用车辆反应速度、行人新路线、
  信号灯新计划；分支点以前的位置与事件不变。
- 分支比较与碰撞：compare_at(left, right, tick) 静默定位到同一时刻，
  比较位置差、车-行人碰撞与事件/碰撞触发原因。
- 独立重开：分支是完整 Scenario；save_run/load_run 保存分支元数据、
  运行时路线、灯控计划与碰撞日志，不依赖内存中的父场景。

## 用法示例

见 tests/test_replay.py 与 tests/test_branching.py：构造 Scenario 后用
ReplayEngine 推进，position_of / signal_phase 读取状态，fired_event_ids
核对事件记录，compare_at 比较两个分支，save_run / load_run 保存场景版本、
种子、运行状态、事件日志与碰撞日志。

后续接入感知算法时，直接读取引擎当前 tick 的 positions() 与
signal_phases()，并共享同一时钟即可。

## 测试

    python -m pytest tests/ -q

覆盖：同 tick 事件固定顺序、逐步与直跳一致性、暂停/中断、跳回过去
不重复触发、交错车辆行人位置核对、保存/加载后继续运行、同刻变灯与刹车、
行人横穿碰撞、分支中断后恢复、分支独立保存重开。
