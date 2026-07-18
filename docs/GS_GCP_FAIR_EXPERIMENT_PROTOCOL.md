# GS-GCP 公平实验协议

Status: pre-registered execution contract, 2026-07-17.

## 1. 实验目标与阶段顺序

正式比较只回答一个问题：在相同 RGB 影像、相同相机、相同 SfM 初始值、
预注册的方法原生训练配方与透明报告的资源预算、以及相同 GCP 评测协议下，不同 Gaussian geometry 方法的绝对几何
精度、稳定性、覆盖率和效率有何差异。

执行顺序固定为：

1. 冻结并验证 v1.3.0 release、split、相机和训练视图；
2. 对每个方法完成 3K 全流水线 qualification；
3. 仅通过 qualification 的方法进入 5K、10K、20K、50K、100K；
4. seed-0 主实验完成后，再进行预注册的随机种子和控制点数量敏感性分析；
5. 所有表格由 CSV/JSON 自动复算，不人工抄写指标。

3K qualification 包含环境构建、训练、固定 checkpoint、metric-depth adapter、
packet 导出、相机兼容性、release-mode evaluator 和独立指标复算。高 RMSE 是方法
结果，不是删点、调参或停止资格测试的理由。

## 2. 统一数据与 split

- 正式 primary track 使用同一个 GS-GCP v1.3.0 release root digest。
- 六场景训练图像列表、图像字节、COLMAP `cameras.bin`、`images.bin`、
  `points3D.bin` 和初始 PLY 均按 SHA-256 锁定。
- 所有方法使用同一场景的完整训练图像列表，不自行删图或加入额外图像。
- control/checkpoint split 对所有方法完全相同：

| Scene | Controls | Checkpoints | Total |
|---|---:|---:|---:|
| 3K | 5 | 4 | 9 |
| 5K | 6 | 4 | 10 |
| 10K | 6 | 4 | 10 |
| 20K | 6 | 4 | 10 |
| 50K | 12 | 11 | 23 |
| 100K | 13 | 12 | 25 |
| Total | 48 | 39 | 87 |

- split 由 surveyed XYZ、空间/高程覆盖、标注数量和视角多样性生成，禁止读取
  模型 residual、depth、alpha、variance 或 scatter。
- GCP 坐标、角色和观测不得参与训练 loss、checkpoint 选择、early stopping 或
  超参数选择。

## 3. 统一 SfM 与相机协议

- Primary track 使用 benchmark 提供的 undistorted PINHOLE images/cameras。
- 每个方法读取相同 COLMAP 外参、内参和初始稀疏点云；不得重新运行 SfM 后替换
  相机或初始点云。
- 默认固定相机位姿。若方法原生包含 pose optimization，primary 结果必须关闭；
  需要研究时另设明确标记的 camera-optimized diagnostic track。
- 不允许 crop、pad、EXIF transpose、重新编号或图像域替换。任何方法特有预处理
  都必须生成逐图像映射、尺寸、内参和哈希，并证明 ray equivalence。
- 标注链固定为 raw pixel -> normalized ray -> benchmark undistorted pixel ->
  method packet pixel。所有坐标使用 zero-based pixel centers。

## 4. 统一分辨率协议

Primary training resolution 名称为
`graphdeco_rminus1_1600_width_cap_v1`，严格复现原版 3DGS `-r -1`：

```text
if original_width <= 1600:
    loaded_width  = original_width
    loaded_height = original_height
else:
    scale         = original_width / 1600.0
    loaded_width  = int(original_width / scale)
    loaded_height = int(original_height / scale)
```

`int` 是 Python 对正数向零截断；宽高分别计算；尺寸来源是 benchmark undistorted
图像实际解码矩阵。该规则不放大小图。真实 3K 示例：`5654 x 4098 -> 1600 x
1159`。

原版 3DGS 直接使用 `--resolution -1`。其他方法不得把各自的 `-1`、R8 或 nominal
scale 当作等价证明，而必须在训练前输出逐图像 loaded width/height、内参和 image
tensor/hash probe，并与同一 canonical dimension manifest 比较。若方法 loader 无法
精确匹配，则使用只读 canonical resized input adapter；adapter 的重采样实现、版本和
像素哈希需预先冻结。不得混用全分辨率、R8、1600 长边或 1600 短边规则。

## 5. 方法参数与训练预算

- 原版 3DGS 使用官方 30K schedule、seed 0 和 iteration 30000 checkpoint。
- 其他方法使用其正式论文/官方代码针对自定义场景推荐的 schedule；每个方法的
  recipe 必须在查看 formal GCP residual 前登记。
- 主表优先保持方法原生优化设计，不强制所有方法拥有相同迭代次数；同时报告
  wall time、GPU-hours、峰值显存和实际迭代/采样预算。
- 禁止以 GCP 误差调节 learning rate、densification、regularization、partition、
  checkpoint 或 early stopping。
- 方法若需要外部 monocular depth/normal/pointmap，必须固定模型、权重哈希、输入
  图像和生成命令，并单独报告额外预处理成本。

## 6. 无侵入指标探测

训练代码只允许通过外部进程监控、日志解析、只读 checkpoint 检查或显式
evaluation-only adapter 采集指标。不得把 GCP evaluator、depth packet 或监控逻辑
接入训练 loss/autograd graph。

每个 run 至少记录：

- Git commit/tree、dirty status、完整命令和环境 lock；
- GPU、CUDA、PyTorch、编译器、峰值显存、wall time；
- checkpoint 大小、Gaussian 数量和模型 tree hash；
- 训练图像/相机/初始点云 hashes；
- packet export 时间、packet 数量/大小/hash 与失败原因；
- 渲染质量的 PSNR/SSIM/LPIPS 仅在统一 image holdout 可定义时作为 secondary；
- 正式 geometry 指标始终来自冻结 GCP evaluator。

## 7. 正式几何指标

Formal P1 depth 固定为 `alpha_normalized_expected_camera_z = M1 / A`，语义为
camera-z。主表按场景报告 checkpoint：

- RMSE-H、RMSE-Z、RMSE-3D；
- median、p90、max 3D residual；
- valid checkpoint/control 与 observation coverage；
- failure count/reason；
- per-point residual、view count、aggregation mode、multiview scatter；
- Sim(3) scale/rotation/translation、control residual 和 conditioning；
- 训练/导出/评测耗时、峰值显存和模型大小。

跨场景汇总使用 scene macro average；pooled micro average 仅作 secondary。不同方法
coverage 不同时，必须同时报告失败率，不能只比较 surviving-point RMSE。

Harmonic camera-z、expected inverse depth、variance、alpha、surface/mesh depth 和
沿 ray 分解均为 diagnostic，不得替代 formal P1 或用于选择方法参数。

## 8. 方法准入与失败分类

只纳入已有正式会议或期刊论文、官方公开代码和固定 commit 的方法。每个方法先过
3K；仅 arXiv、无法恢复官方实现或无法满足 camera-z contract 的方法不进入正式
矩阵。

- `PASS`: 完整链可复算并满足所有 frozen gates；
- `METHOD_FAILURE`: 方法训练或几何质量失败，但协议正常；
- `PROTOCOL_INCOMPATIBLE`: 无法提供相同相机/packet/P1 语义；
- `BLOCKED`: 代码、license、权重或关键 artifact 不可恢复。

## 9. 隔离与不可覆盖运行

每个 method 使用独立 fixed-commit worktree、environment、CUDA build cache、run root
和临时目录。数据与 release 只读；训练输出不得写入源码或数据目录；run root 已存在
即失败。正式启动前后均验证数据 tree digest 和代码 clean status。

新运行命名空间统一使用 `gs-gcp-v13`。具体合同见
`configs/gcp_v13_workspace_isolation_v1.json`。
