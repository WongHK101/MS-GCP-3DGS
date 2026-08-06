# GS-GCP 公平实验协议

Status: clean-R4 execution contract frozen and externally accepted on 2026-08-07; formal 3K training waits only for deployment/runtime preflight and GPU availability.

## 1. 目标与准入顺序

本 benchmark 比较大场景、几何类 Gaussian Splatting 方法在相同 RGB、相同相机、相同共享 SfM 初始化和相同 GCP 测量协议下的绝对几何精度、覆盖率、稳定性、效率与渲染质量。

固定执行顺序为：

1. 校验 v1.3.0 release、训练源、R4 输入、RGB holdout 和相机身份；
2. 每个方法先完成 3K 全流水线 qualification；
3. 只有通过 3K qualification 并完成外部审核的方法才进入其余五个场景；
4. 主实验使用 seed 0；敏感性分析仅在主实验后按预注册方案执行；
5. 所有正式表格从 CSV/JSON 自动复算，不人工抄写指标。

3K qualification 包含输入验证、环境构建、从零训练、固定 checkpoint、held-out RGB、metric-depth packet、相机兼容性、release-mode GCP evaluator 和独立复算。高 RMSE 是结果，不是调参、删点或提前停止的理由。

## 2. 冻结数据、RGB holdout 与 SfM

- release root digest 固定为 `513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75`。
- RGB holdout 是 image-loss-held-out、pose-known 协议；每场景测试图数量为 `ceil(N/8)`。
- 冻结视图数如下：

| Scene | Full | Train | Test |
|---|---:|---:|---:|
| 3K | 94 | 82 | 12 |
| 5K | 101 | 88 | 13 |
| 10K | 976 | 854 | 122 |
| 20K | 298 | 260 | 38 |
| 50K | 2208 | 1932 | 276 |
| 100K | 2510 | 2196 | 314 |

- 共享初始化允许使用冻结 all-image SfM 的 point XYZ、RGB 和初始 PLY。
- 训练只允许读取 train RGB、train intrinsics/extrinsics 和共享初始点；禁止读取 test RGB、test 2D tracks、test visibility、GCP 身份/残差和 survey coordinates。
- primary track 固定相机位姿；方法原生 pose optimization 若需研究，只能进入单独标记的 diagnostic track。

## 3. Primary R4 分辨率协议

唯一 active 规则是 `graphdeco_quarter_resolution_v1`：

```text
loaded_width  = round(decoded_width  / 4)
loaded_height = round(decoded_height / 4)
```

`round` 是 Python ties-to-even，宽高分别计算；尺寸来自 Pillow 解码后的 undistorted RGB 矩阵。禁止 crop、pad、EXIF transpose 或重编号。示例：`5654 x 4098 -> 1414 x 1024`，`5658 x 4099 -> 1414 x 1025`。

公共输入由 `configs/gs_gcp_r4_input_materialization_v1.json` 定义：Pillow 11.1.0 使用 `Image.resize(size)` 默认 BICUBIC，输出无损 RGB PNG；PINHOLE 内参按实际 R4 宽高分别缩放，外参保持不变；逐图记录源 SHA、PNG SHA 和 RGB uint8 SHA。

各方法必须证明输入 tensor、相机和 ray 与该公共 R4 域一致。不能把方法自己的 `-1`、R8 或 nominal scale 当作等价证明。

旧 `graphdeco_rminus1_1600_width_cap_v1` 及其 1600-width 结果仅保留为历史诊断证据，不属于 active primary track。

## 4. 干净官方 3DGS 基线

- 官方源码固定在 commit `2eee0e26d2d5fd00ec462df47752223952f6bf4e`、tree `5eee127dfc0942bf83d9fdd72328e03ec0cbf6c4`。
- 训练源码必须保持 clean；不允许 path-backed loader、runtime serializer patch 或旧 checkpoint resume。
- benchmark 先生成物理隔离的 `train/` 和 `test/` 根目录。两个 COLMAP 子模型都移除 POINTS2D tracks；训练根中不存在 test 图像或 test camera record。
- 官方训练只绑定 `train/`，使用 `--resolution 1 --iterations 30000`、seed 0、官方默认 schedule 和 iteration 30000 checkpoint；`--eval` 关闭，因为 holdout 已由 benchmark 物理隔离。
- R4 PNG + 缩放相机必须与“冻结全分辨率 JPEG + 官方 `--resolution 4`”在像素、FoV 和 normalized ray 上等价。
- 唯一 active recipe 是 `configs/gs_gcp_v13_original_3dgs_recipe_v3.json`。

旧 1600-width、path-backed、serializer-modified 路线的配方、checkpoint、结果和 qualification 不得继承到本路线。

## 5. 训练预算与反泄漏

- 每个方法的 recipe 必须在查看 formal GCP residual 前冻结。
- 不强制所有方法使用相同迭代次数；使用正式论文或官方代码对自定义场景推荐的固定 schedule，并报告实际迭代/采样预算。
- 禁止用 GCP residual 选择 learning rate、densification、regularization、partition、seed、checkpoint 或 early stopping。
- 外部 monocular depth/normal/pointmap 若为方法必要组成，必须冻结模型、权重 SHA、输入和生成命令，并单独报告成本。

## 6. 测量协议

RGB 指标在 test 图像上计算 `PSNR / SSIM / LPIPS-VGG`，聚合为 per-image → scene mean → six-scene macro mean。

正式几何张量固定为：

```text
alpha_normalized_expected_camera_z = M1 / A
```

语义固定为 camera-z。每场景报告 RMSE-H、RMSE-Z、RMSE-3D、median、p90、max、有效 control/checkpoint coverage、failure reason、Sim(3) 和资源开销。跨场景 primary 汇总使用 scene macro；pooled micro 仅为 secondary。

不存在强制 composite leaderboard score。surface 方法先验证同一 `M1/A camera-z` 公共轨道；只有确实无法兼容时，才另设清楚标记的 native secondary track，而不是预设强制双轨。

## 7. Qualification 与场景顺序

当前 clean-R4 合同审核状态是 `CLEAN_R4_CONTRACT_PASS`，3DGS qualification 状态仍是 `NOT_RUN`，full matrix 尚未解锁。3K 通过且其结果证据包再经外部接受后，复用该次 3K 结果，并按以下顺序运行：

1. 100K
2. 50K
3. 20K
4. 10K
5. 5K

最大场景预检发现共享协议失败时停止全矩阵；资格通过后的单场景独立方法失败记录后继续下一场景。

## 8. 隔离、硬件与失败记录

- 每个 method 使用独立 fixed-commit worktree、environment、build cache 和 run root。
- 原始数据、release 和物化输入在训练时只读；run root 已存在即失败，不覆盖。
- 每次启动重新记录单 GPU UUID、硬件、驱动、CUDA、PyTorch、命令、环境 lock、源码 clean 状态和资源峰值；不继承旧 GPU UUID。
- 正式失败按阶段记录，包括 camera load、initialization、optimization、serialization、render、packet export 和 formal evaluation；失败不能被静默删去。

## 9. 当前权威文件

- 方法注册表：`configs/gs_gcp_method_registry_v1.json`
- R4 规则：`configs/gs_gcp_quarter_resolution_v1.json`
- R4 物化合同：`configs/gs_gcp_r4_input_materialization_v1.json`
- RGB split：`configs/gs_gcp_rgb_holdout_split_manifest_v1.json`
- 测量套件：`configs/gs_gcp_common_measurement_suite_v1.json`
- 3DGS recipe：`configs/gs_gcp_v13_original_3dgs_recipe_v3.json`
- full-matrix gate：`configs/gs_gcp_v13_original_3dgs_full_matrix_v2.json`
