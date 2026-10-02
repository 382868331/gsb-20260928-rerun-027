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
  以前的轨迹、事件与碰撞（分支点后的父场景事件不带入分支），之后可替换
  车辆/行人后续路线、信号灯计划，或设置车辆加减速/刹车反应参数。
  分支定义完整写入 Scenario，可单独保存并独立重开；向后跳转重放不会
  重复累计分支点之前的事件。
- 碰撞与原因：每 tick 按参与者碰撞半径去重产生 CollisionRecord；
  事件 action 支持 reason，compare_at(a, b, t) 可在同一时刻比较位置、
  碰撞集合/原因与事件触发原因。
- 差异定位：first_divergence(a, b, until_ms) 逐 tick 对比两个分支，
  返回第一处行为差异（时刻、维度与涉及的车辆/信号灯）。
- 反应延迟分支：BranchChanges.vehicle_reaction_delays 为选定车辆设置
  `signal_red`、`signal_green`、`lead_brake` 等刺激的延迟（毫秒）。
  PerceptionRecord 记录感知时刻与事实，ResponseRecord 记录原定触发/到期
  时刻、动作及 `scheduled/executed/superseded/cancelled/ignored` 状态；
  待执行动作保存并恢复后仍在原到期时刻执行，回跳与再次分支不会重复执行。
- 反应方案比较：first_response_comparison 比较首感和首响应，
  minimum_gap_between 统计最小间距，conflict_event_chain 汇总冲突前的
  感知、响应和脚本事件链。
- 不确定感知对照：UncertainPerception 描述"驾驶员在
  [earliest_ms, latest_ms] 内看到前车制动/信号切换"这类区间事实；
  evaluate_uncertainty 枚举区间内候选感知时刻（可用 samples 只考察
  边界），每个组合作为独立分支运行，SchemeResult 保留该方案实际采用的
  感知时刻与反应延迟。PerceptionConstraint 约束同一车辆多条感知范围
  的关系（order 先后 / same_observation 同次观察），不满足约束的组合
  不会生成。报告把已发现冲突的方案（conflicts）、已考察无碰撞的方案
  （clean）与未考察区间（unexplored）分开列出；coverage 恒为
  "discrete"，is_continuous_safety_verified 恒为 False——离散对照
  不能被展示成未经验证的连续安全结论。append_scheme 在已有冲突附近
  追加具体方案且不回写旧方案的感知事实；explain_difference 说明差异
  来自感知选择、反应延迟还是后续信号安排，并给出首次分开的 tick 与
  该时刻的交通事件。计划感知（PlannedPerception）随场景存档，
  覆盖同一 (vehicle, stimulus, source) 的自动感知；退出车辆的待响应
  事件不进入派生方案，感知已发生而动作未响应时可 save_run 后继续。

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
行人横穿碰撞、分支中断恢复、分支对比与分支存档独立重建、分支不带入
父场景后续事件、同时刻多分支对照与首处差异定位、分支内再分支的事件归属。
不确定感知：边界时刻采样与未考察区间列出、感知范围约束（先后/同次观察）、
同刻刺激、嵌套分支保留父方案感知事实、追加方案不回写旧方案、退出车辆
待响应不进入派生方案、感知与响应之间保存继续、差异归因与首次分开定位。
