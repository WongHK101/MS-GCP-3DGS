# M3M-GCP 20K 十方法 RGB/GCP 执行方案 v1

状态：`REVIEW_CANDIDATE`  
场景：`gcp_20000_20260602`  
范围：训练、GCP 几何评测、held-out RGB 评测和资源统计；LiDAR 明确不在本轮范围内。  
协议：继续使用 `m3m_gcp_native_quarter_geometry_v2`，本文件是场景执行方案，不创建新的几何协议版本。

## 1. 目标与简化原则

20K 是三场景论文方案中的中尺度、低影像密度核心场景。本轮目标是在不改变各方法既定算法预算的前提下，尽可能完成十方法的可比结果，同时把 3K 和 100K 已经暴露的问题一次性吸收到公共执行链。

- 只做一次启动前公共预检、一次连续批量执行和一次集中终审。
- 不做逐方法外部审核，不为每个方法设置一次性门票或重复 micro-parity。
- 不做独立 pilot；实际正式运行的前 200 个迭代/首个阶段同时承担健康检查。
- 真实 OOM、数值失败、官方流程不适配或预算超限如实记录并继续下一方法，不降低分辨率、减少正式迭代、改损失或按结果调参来“救活”。
- 公共路径、环境、清单或评测器接线错误允许最小修正；优先复用已有模型/包，禁止因此重训。

## 2. 数据与真值绑定

正式输入：

- 发布根：`M3M-GCP-colmap-native-quarter-v1`
- 场景：`gcp_20000_20260602`
- 影像：298 张，其中训练 260、held-out 38
- 分辨率：1414×1025；输入已是 COLMAP native-quarter，训练统一使用 `resolution=1`，不得再次 `/2`、`/4`、resize、crop、pad 或重编码
- 输入 manifest 文件 SHA-256：`8dc444f767f5a65bc8612eacd1527077a9dcd457f7fe04ebd8bf2f683be71fa7`
- 输入 manifest canonical SHA-256：`1d305e78974c299c7e63cc7774ea6e63ac3f428511d5b85b4dd252790cd2e64c`
- SfM 语义：先用全部 298 张影像完成共同 COLMAP/SfM，再划分训练与 held-out；训练损失和训练先验只能读取 260 张训练影像，但允许使用这套共同 SfM 的稀疏点几何
- 训练、held-out 的图像记录必须互斥；每个方法正式启动时 loader 必须报告 260 train、0 held-out

20K GCP overlay：

- 正式 checkpoint：G30、G31、G37、G38，共 4 个
- control：G28、G29、G33、G35、G36，共 5 个，仅用于公共 Sim(3) 和诊断
- `dyl2`：继续隔离为 diagnostic-only，不得重新进入正式集合
- 正式观测：116 条（checkpoint 51、control 65），覆盖 103 个唯一影像视图；nadir 68、oblique 48
- 所有方法共享同一冻结 Sim(3)，不得用方法输出、checkpoint 或 LiDAR重新配准

## 3. 方法池与训练路线

单 seed：0。GOF 继续作为历史退役方法，不扩展到 20K。

| 顺序 | 方法 | 20K 路线 | 正式预算 | 先验边界 |
|---:|---|---|---:|---|
| 0 | 3DGS | 复用已完成的 20K 官方训练模型；不重训 | 30K 已完成 | RGB+COLMAP |
| 1 | 2DGS | 复用 3K 已验证官方路线 | 30K | RGB+COLMAP |
| 2 | PGSR | 复用 3K 路线和 100K 已验证输入接线 | 30K | RGB+COLMAP |
| 3 | RaDe-GS | 复用 3K 路线和 100K 已验证输入接线 | 30K | RGB+COLMAP |
| 4 | SOF | 复用 3K 正式路线 | 30K | RGB+COLMAP |
| 5 | QGS | 复用 3K 正式路线 | 30K | RGB+COLMAP |
| 6 | GSPrior | 复用已验证归一化训练域和测试相机域转换 | 40K | 仅用训练渲染生成场景内部 TSDF；无外部真值 |
| 7 | CityGS-X | 复用 3K/100K 成功路线 | 100K | DAv2-L，仅生成 260 个训练视图先验 |
| 8 | CityGaussianV2 | 官方 aerial coarse/fine 路线，预先固定 2×2 分块并合并 | coarse 30K + 每块 fine 60K | DAv2-L，仅训练视图；2×2 由 100K 4×4 的近似每块覆盖面积预先推导，不看结果选块数 |
| 9 | MetroGS | 复用 3K/100K 已验证 Pi3-Align + MoGe-2 路线 | 150K effective image iterations，即 37.5K optimizer steps、batch 4 | Pi3/MoGe 只读 260 个训练视图 |

CityGaussianV2、CityGS-X 等需要 tracks 的方法，使用“260 个训练 image records + 全量 298-view SfM 产生的原始 points3D tracks”的专属输入；不得把 held-out image records 或 held-out RGB 放入训练/先验根。所有外部先验的权重、源码版本和命令沿用已验证配置，不因 20K 结果调整。

## 4. 一次性公共预检

901 开机后只执行一次：

1. 验证 manifest 双哈希、298/260/38 数量、train/test 名称互斥、1414×1025 解码尺寸和全量 SfM 后划分语义。
2. 验证十方法源码提交、现有环境、权重与公共评测代码可读；禁止重新配置已经可用的环境。
3. 验证方法专属输入生成器不含硬编码的 `82/12`、`2196/314` 或 `1024` 高度；所有数量从本场景 manifest/allowlist 读取。
4. 验证 3DGS 已完成模型存在、正式迭代为 30K、文件可读并记录 SHA-256；模型路径为 `/root/autodl-tmp/runs/m3m-gcp-native-quarter/3dgs-original/gcp_20000_20260602/seed0-30k-20260825T121655Z/model`。
5. 验证 GCP overlay 为 4 checkpoints、5 controls、116 条正式观测、103 个唯一评测视图，`dyl2` 已隔离。
6. 启动前可用磁盘建议不少于 300 GiB；低于 150 GiB 时暂停新方法并先做精确、可恢复的临时产物清理。

预检通过后直接进入连续批次，不再为每个方法重复静态审计或单独 smoke。

## 5. 执行与故障处理

- 同一时刻只允许一个 GPU 训练/渲染进程。
- 先对现有 3DGS 模型跑通 20K GCP 和 RGB 端到端评测；它只验证公共执行链，不用于按结果修改其他方法配方。
- 随后按表中顺序训练剩余九方法。每个方法训练完成后立即记录模型哈希和资源，渲染 GCP/RGB 所需视图；CPU 指标计算可在不争用训练 GPU、内存和磁盘带宽的条件下与下一方法重叠。
- 方法专属先验 just-in-time 生成；成功完成模型和证据哈希后再删除可重建的中间先验，不删除最终模型。
- 公共/基础设施错误：错误发生在 optimizer 启动前，或能证明为路径、环境、manifest 标签、硬编码视图数、公共 exporter/evaluator 接线问题时，允许最小修正并重试；不计作算法失败。
- 外部断网/关机：若官方 checkpoint 能保持同一预算与状态则恢复，否则允许从新目录重新开始一次并标记基础设施重启；不得改变配方。
- 算法错误：optimizer 启动后出现真实 CUDA OOM、非有限训练、官方算法异常或不合理的资源扩张时，记录 `OOM` / `ALGORITHM_FAILED` / `BUDGET_EXCEEDED`，保存轻量证据并继续。禁止通过降低输入分辨率、减少迭代、改 batch 语义、裁点或调参补结果。
- 方法专属 exporter 接线错误只修 exporter 并复用模型；统一评测器错误只重算受影响评测，不重训任何方法。

## 6. 非 LiDAR 正式评测

### 6.1 GCP 几何主轨

- 使用现有公共 `A/M1` expected-camera-z 算子、float64 双线性插值、同一覆盖门槛和同一场景 Sim(3)。
- 每方法渲染固定的 103 个 GCP 观测视图，不按方法补相机。
- 主排名：4 个 checkpoint 的 RMSE-3D；同时报告 RMSE-H、RMSE-Z、median、P95、max、逐点覆盖和 ±0.5 px 敏感性。
- 5 个 controls 仅作对齐/泛化诊断；`dyl2` 不进入正式统计。
- 4/4 checkpoints 全部通过覆盖才记 `COMPLETE_RANKED`；否则为 `INCOMPLETE_UNRANKED`，已成功子集仅诊断。

### 6.2 held-out RGB 轨

- 使用全部 38 张冻结 held-out 图像、原始 1414×1025 像素域、全帧、无 mask/crop/resize。
- 统一报告 PSNR、SSIM、LPIPS-VGG；背景沿用各方法冻结训练背景（当前均为黑色）。
- held-out RGB 不得参与 checkpoint、曝光、外观或超参数选择；所有 38 视图完成才给正式场景均值。

### 6.3 资源与状态

记录训练/先验/渲染墙钟时间、峰值显存、峰值主存、GPU 型号、最终模型大小、可得时的 Gaussian/primitive 数、OOM/失败类别。能耗若采集可靠则作补充列，缺失不阻断精度结果。

LiDAR 不导出、不评测，也不作为本批训练或验收前置条件。新的逐场景 DJI Terra 点云到位后，以保留的最终模型另开独立评测批次。

## 7. 存储与归档

- 成本控制遵循 `M3M_GCP_AUTODL_COST_CONTROL_CONVENTION_V1.md`：最后一项服务器实验/评测完成并完成最小落盘核验后立即普通关机；本机 Excel、论文表格和集中审核不得延迟关机。
- 所有最终模型留在 901，后续 LiDAR 评测必须可直接复用。
- 成功后保留最终 checkpoint/merged model、配置、命令、manifest、资源日志、逐视图轻量指标和总指标。
- 原始 GCP/RGB 大包、重复渲染、非正式中间 checkpoint 和可重建 prior 仅在最终模型与指标完成验哈后清理。
- 失败方法保留 failure receipt、stderr/stdout 尾部、资源/OOM证据和必要的最后状态；大体积无效 scratch 精确清理。
- 本机只拉回轻量证据、核心指标、少量论文候选可视化和更新后的 Excel；不拉回全部模型。

预计剩余九方法约 48–72 GPU 小时，连同评测和串行衔接约 2–4 天；瞬时工作空间预计 150–250 GiB。该估计不是提前终止阈值。

## 8. 审核与验收

- 启动前：仅审核本方案的数据绑定、方法路线、评测口径和故障边界；不重审已经通过的 3K/100K源码、权重、数学算子和 LiDAR。
- 执行中：不逐方法暂停审核。真实方法失败不是红线，自动继续；只有公共数据错绑、held-out 泄漏、统一评测器构念变化或磁盘安全线才暂停整批。
- 批次结束：做一次集中终审，核对十方法状态、模型身份、38-view RGB 完整性、4-checkpoint GCP 完整性、资源记录和清理结果；最多提出 3 项必须修正。
- 终审后更新本机总表和论文 Table1–TableN 工作表。LiDAR 列保持 `PENDING_NEW_TERRA_REFERENCE`，不得引用旧裁剪数值。
