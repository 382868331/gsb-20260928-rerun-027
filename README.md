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
- 任意点分支：branch_from(engine, id, BranchChanges(...), at_ms) 保留分支点
  以前的轨迹、事件与碰撞，之后可替换行人路线、信号灯计划，或设置车辆
  加减速/刹车反应参数。分支定义完整写入 Scenario，可单独保存并独立重开。
- 碰撞与原因：每 tick 按参与者碰撞半径去重产生 CollisionRecord；
  事件 action 支持 reason，compare_at(a, b, t) 可在同一时刻比较位置、
  碰撞集合/原因与事件触发原因。

## 用法示例

见 tests/test_replay.py：构造 Scenario 后用 ReplayEngine 推进，
position_of / signal_phase 读取状态，fired_event_ids 核对事件记录，
save_run / load_run 保存场景版本、种子、运行状态与事件日志。

后续接入感知算法时，直接读取引擎当前 tick 的 positions() 与
signal_phases()，并共享同一时钟即可。

## 测试

    python -m pytest tests/ -q

覆盖：同 tick 事件固定顺序、逐步与直跳一致性、暂停/中断、跳回过去
不重复触发、交错车辆行人位置核对、保存/加载后继续运行、同刻变灯与刹车、
行人横穿碰撞、分支中断恢复、分支对比与分支存档独立重建。
