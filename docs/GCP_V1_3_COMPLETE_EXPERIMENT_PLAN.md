# MS-GCP v1.3 完整协议冻结与多算法实验计划

Status: execution plan, not yet a frozen release, 2026-07-17.

## 1. 当前结论与启动边界

当前数据审计已经达到“可进入协议冻结实现”的状态，但尚未达到“可直接批量跑正式实验”的状态。

已完成并可作为冻结输入的事实：

- 六场景 working annotations 共 1,383 条，其中 1,155 条为 Good；
- 状态/坐标错误 0，跨点位同图近邻碰撞 0；
- 50K `DJI_20260610161948_0002_D.JPG` 保持正常训练影像；
- 该图上的 G33、G39 两条模糊观测为 Ambiguous，仅按 Good-only 规则退出评测；
- 无 image-level exclusion，无 GPS 特殊筛选或特殊论文叙事；
- authoritative RTK 坐标使用 corrected package；
- residual-blind geometry-only split 已两次独立生成且 12/12 文件 byte-identical；
- 六场景候选 split 均为 `ready_candidate_not_frozen`。

正式 control/checkpoint 数量候选：

| Scene | Formal points | Controls | Checkpoints |
|---|---:|---:|---:|
| 3K | 9 | 5 | 4 |
| 5K | 10 | 6 | 4 |
| 10K | 10 | 6 | 4 |
| 20K | 10 | 6 | 4 |
| 50K | 23 | 12 | 11 |
| 100K | 25 | 13 | 12 |
| Total | 87 | 48 | 39 |

尚未完成、因此阻止“正式冻结”声明的事项：

1. transactional v1.3 canonical release；
2. 1,383-row raw-pixel/camera/projection provenance；
3. v1.3 payload manifest 与 root digest；
4. evaluator v1.3 release loader 和 real-release tests；
5. common method-independent training contract；
6. original 3DGS 的 3K end-to-end reference smoke。

在以上六项通过前，不启动多方法六场景批量训练。可以先做 no-GPU adapter/source audit。

## 1.1 方法工作区与原始数据隔离硬约束

- 原始影像、COLMAP sources、RTK、annotations 和正式 release 对算法进程只读；
- 每个方法使用独立 clean fixed-commit worktree，不在代码目录编译 CUDA 或写日志；
- 每个方法使用独立锁定环境，不允许 global/user-site `pip` 或复用其他项目环境；
- 每个 method/run 使用独立 `TORCH_EXTENSIONS_DIR`、build root 和 temp root；
- 每个 method/scene/run 使用唯一不可覆盖 run root，内部固定分为 preflight、training、checkpoints、packets、evaluation、diagnostics、audit；
- 训练、模型、packet、评测和临时文件不得写入 dataset/release/code roots；
- 正式 run 前后核对 source inventory/hash；任何原始输入变化使该 run 失效；
- 所有 launch 必须先通过 `validate_gcp_v13_workspace_isolation.py`，不能只依赖人工遵守。

完整规则见 `docs/GCP_V1_3_METHOD_WORKSPACE_ISOLATION_POLICY.md`。

## 2. Phase F: v1.3 协议冻结

### F0. 冻结输入快照

- 固定六个 current working annotation CSV 的 SHA-256；
- 固定 1,383-row canonical spine，保留 Good/Ambiguous/Not visible 全部审核状态；
- `formal_eligible=true` 仅允许 `visible=1 && quality=good` 且 raw 坐标 finite/in-bounds；
- Ambiguous 与 Not visible 保留为 provenance，禁止静默删除或改写为 Good；
- 固定 authoritative RTK package、87 个 formal points 和 48/39 split identities；
- G47 不在正式候选池内，未经坐标纠正不得加入；
- 固定每场景完整 training image list；`0002` 正常保留。

Gate F0：所有输入 hash 唯一、worktree clean、split 两次生成 byte-identical、无未决人工复核项。

### F1. 生成 transactional release v1.3.0

- 在同一文件系统唯一 staging 目录生成，全部通过后原子 rename；
- 正式目录存在时 hard fail，不覆盖、不合并；
- 每条 observation 保存稳定 ID、raw image SHA、RGB orientation hash、raw dimensions、raw coordinates 和 QC status；
- 标注域固定为 decoded raw image、0-based pixel centers、EXIF orientation ignored/no transpose；
- 重算 raw pixel -> normalized ray -> undistorted benchmark target pixel；
- 保存 source/target cameras/images hashes、per-camera hashes、per-pose hashes 和 unique image-level mapping records；
- 611 条 v1.2.2 row lineage 可验证；新增 observation 使用相同 deterministic ID/serialization contract；
- point table、split、scene metadata、RTK provenance、low-light metadata 纳入 payload；
- 生成 payload manifest、root digest、detached hashes；重复 staging 生成必须 byte-identical。

Gate F1：1,383/1,383 rows preserved；1,155 formal-eligible rows；ID duplicate=0；projection/round-trip/camera/orientation/integrity tests 全通过。

### F2. evaluator v1.3 release-mode support

- schema -> layout 显式映射；未知 schema hard fail，不回落旧 raw-coordinate CSV；
- evaluator 从 canonical raw pixel 重算 target projection，不直接信任 cached target coordinate；
- release-owned annotation/GCP/split/metadata 禁止 CLI override；
- Good-only eligibility 在 loader 中验证，不能由运行命令重新定义；
- `control_policy=require_all`，control/checkpoint leakage hard fail；
- `min_valid_observations=1` 保持 formal aggregation 规则；每个点同时报告 view count 和 robust-multiview eligibility；
- release manifest、root digest、camera/pose/mapping/orientation hashes 均为 runtime hard gates；
- v1.2.2 loader tests 继续通过，v1.2.2 保持只读历史 diagnostic release。

Gate F2：synthetic negative tests + real six-scene v1.3 interface smoke；1,383/1,383 rows validated；不读取 depth、不计算 metric。

### F3. common training/camera contract

- benchmark camera track 为唯一 primary track；所有方法使用相同 undistorted PINHOLE images/cameras；
- formal primary resolution 固定为 R8；实际 width/height、rounding、intrinsics 和 pose 逐 view 写 manifest；
- 不允许 crop/pad、相机重优化或 method-specific pose，除非另行审核 adapter/remap；
- pixel convention 固定为 `zero_based_pixel_centers`；
- 所有方法使用完全相同的 scene image list，包含 `0002`；
- GCP pixels、survey XYZ、control/checkpoint roles 不进入训练、checkpoint selection、early stopping 或超参数选择；
- 每方法 recipe 在首个训练前冻结：official repo commit、environment、seed、iteration/budget、densification、checkpoint rule、resolution 和 launch command；
- 原版 3DGS primary recipe 使用 official 30K schedule；其他方法采用论文/正式代码推荐 schedule，但必须预注册，不能根据 GCP residual 调参；
- primary seed 固定为 0。核心方法完成后，在 3K 追加 seeds 1/2 作为随机性 sensitivity，不替代 seed-0 primary result。

Gate F3：六场景 image/camera manifests 固定；每方法 recipe 的输入和 checkpoint rule 在看到 formal residual 前登记。

### F4. common metric-depth packet contract

Formal P1 固定为：

`alpha_normalized_expected_camera_z = M1 / A`，semantics=`camera_z`。

每个 adapter 必须输出 packet v2：

- `A=accumulated_alpha`；
- `M1=weighted_camera_z_sum`；
- `M2=weighted_camera_z_second_moment`；
- `H=weighted_inverse_camera_z_sum`；
- expected camera-z、expected inverse camera-z、harmonic camera-z；
- raw camera-z variance、valid mask；
- historical invalid unnormalized inverse payload，仅作 diagnostic。

固定数值协议：

- alpha cutoff `1/255`；
- early termination `1e-4`；
- numerical support floor `1e-6`；
- normalization epsilon metadata `1e-12`；
- variance clamp tolerance `1e-6`，不修改 raw NPZ；
- variance validation abs floor `1e-5`、ULP factor `8`、rtol `0`；
- packet patch `native_packet_pixel_patch_v1`，size `7`、radius `3`；
- raw negative variance只在验证后生成 diagnostic view，不能影响 formal P1。

每个方法 adapter 必须通过 tiny synthetic GPU parity、raw A/M1/M2/H CPU reference、multi-opacity、off-axis camera-z、zero-alpha、two-layer、backward compatibility 和 packet recomputation tests。方法自己的 median depth、surface intersection depth 或 mesh depth只能作为附加 diagnostic，不得替代 P1。

Gate F4：adapter 未能证明相同 camera-z compositing semantics 时，不进入 formal ranking。

### F5. original 3DGS 3K end-to-end reference smoke

1. 从 clean official 3DGS commit 训练 3K seed 0；
2. 记录 image/camera/checkpoint/log/environment/content hashes；
3. 只对 v1.3 formal rows 引用的 unique views 导出 packet；
4. 生成 packet-native camera compatibility wrapper；
5. 运行 coordinate-only、ray-equivalence、packet/ref consistency gates；
6. 运行 release-mode evaluator；
7. 独立从 CSV 重算 summary，并核对 JSON/CSV；
8. 输出 checkpoint identities、coverage、RMSE-H/Z/3D、median/p90/max、per-point residual、view count、scatter、Sim(3) 和 conditioning。

3K smoke 的通过标准是协议链完整且可复算，不是误差必须接近某个旧结果。禁止根据 RMSE 调整数据、split、depth semantic 或阈值。

Gate F5：通过后才可将 v1.3 标记为 frozen primary benchmark，并启动其他算法训练。

## 3. 方法清单与接入顺序

只纳入已有正式会议/期刊版本的方法。UMGS 不作为论文方法。

Core formal candidates：

1. original 3DGS；
2. 2DGS；
3. PGSR；
4. RaDe-GS；
5. GOF；
6. CityGaussianV2；
7. CityGS-X；
8. MetroGS。

Conditional 3K feasibility candidates：

9. QGS；
10. GFSGS。

QGS/GFSGS 只有在 3K official-code training、camera contract、packet v2 adapter 和 formal evaluator 全部通过后，才升级为六场景方法。仅 arXiv 方法不增加实验槽位。

## 4. Phase A: 每方法 3K qualification

每个方法按相同顺序执行：

1. official publication/code/license/commit audit；
2. isolated environment build 和 tiny official smoke；
3. v1.3 3K camera/image ingestion audit；
4. seed-0 official recipe training；
5. eval-only packet adapter 实现；
6. CPU/CUDA compositing parity 和 eval-disabled compatibility；
7. annotated-view packet export；
8. compatibility wrapper 与 packet reuse audit；
9. release-mode formal evaluator；
10. independent metric recomputation；
11. run package、checkpoint、packet、source 和 environment hashes。

资格判定：

- PASS：完整 formal chain 可复算，所有 frozen controls 有有效 observation；
- METHOD_FAILURE：训练/几何/coverage 失败，但协议正常，保留失败证据；
- PROTOCOL_INCOMPATIBLE：无法提供相同 P1 或 camera mapping，不进入 ranking；
- BLOCKED：代码/权重/license/artifact 不可恢复，不用替代实现伪装该方法。

高 RMSE 本身不是停止或删点理由。

## 5. Phase B: 六场景正式训练与评测

只有通过 3K qualification 的方法进入。场景顺序固定：3K -> 5K -> 10K -> 20K -> 50K -> 100K。

Core primary matrix 为 8 methods x 6 scenes = 48 seed-0 training runs。QGS/GFSGS 若通过 qualification，最多增加 12 runs。完成 primary 后，core methods 在 3K 追加 seeds 1/2，共 16 个 stochastic-sensitivity runs。

每个 method-scene run 使用唯一、不可覆盖的 run root，并保存：

- exact command、git commit、dirty status、container/environment；
- GPU/CUDA/PyTorch/compiler、GPU model、peak memory、wall time；
- input image/camera/release hashes；
- checkpoint/model tree hash；
- packet manifest、per-packet SHA、compatibility wrapper；
- formal evaluator outputs、console logs 和 failure records。

一个 scene 的模型质量失败不自动阻止其他 scene。若发现共享 adapter、camera 或 metric semantic 缺陷，立即停止该方法后续运行；不得边跑边改协议。

## 6. 正式输出指标

Primary ranking 按 scene 分别报告 checkpoint：

- RMSE-H、RMSE-Z、RMSE-3D；
- median、p90、max 3D residual；
- valid checkpoint/control count；
- valid observation coverage；
- failed point/observation count 和原因。

必须同时公开：

- control residuals 和 control identities；
- per-checkpoint residuals；
- valid view count、aggregation mode、multiview scatter；
- Sim(3) scale/rotation/translation、control conditioning；
- alpha 和 variance diagnostic coverage；
- model size、training time、render/export time、peak GPU memory。

汇总规则：

- per-scene result 为主，不用大场景点数淹没小场景；
- six-scene macro average 为跨场景摘要；
- pooled/micro average 仅作 secondary；
- paired method comparison 以相同 checkpoint identities 为单位；
- bootstrap/permutation 以 point 为采样单位，不把多个 views 当独立 survey samples；
- coverage 不同的方法不能只比较 surviving-point RMSE，必须并列报告失败率。

## 7. 预注册 diagnostics 与 sensitivity

以下均不改变 primary ranking：

1. harmonic camera-z 与 expected inverse camera-z sensitivity；
2. historical invalid unnormalized inverse payload audit；
3. alpha/variance/multiview scatter 与 residual 的关联；
4. along-ray、cross-ray、horizontal、vertical residual decomposition；
5. geometry failure contact sheets：RGB、depth、alpha、variance、annotation crop；
6. 3K seeds 0/1/2 stochastic sensitivity；
7. geometry-only nested control-count sensitivity，仅使用预先生成、residual-blind 的 nested controls；
8. v1.2.2 sparse-control track 作为单独 diagnostic，不与 v1.3 primary 数值混排；
9. v1.1 raw-coordinate结果统一标记 `invalid_due_to_annotation_packet_pixel_domain_mismatch`。

任何 sensitivity threshold、nested split 或代表图必须在读取对应方法 residual 前固定。

## 8. 质量控制与 hard stops

立即停止受影响 run：

- release/root/packet/checkpoint/source hash mismatch；
- camera/image identity、pose、pixel domain、resolution 或 ray equivalence mismatch；
- control/checkpoint leakage；
- missing frozen control 导致无法拟合 formal Sim(3)；
- formal P1 不是 `M1/A camera_z`；
- packet/ref consistency failure；
- 训练读取 GCP residual 或用 checkpoint points 选模型；
- 需要修改 v1.3 pointset/split/annotation/metric threshold 才能继续。

不属于 hard stop：

- RMSE 高；
- floater、几何崩塌、多层高斯；
- 个别 checkpoint 失败；
- 低照度或大场景导致方法质量差。

这些是 benchmark 应记录的模型行为，不能通过删点或调参隐藏。

## 9. 审核包与论文表格

每个阶段生成一个统一审核包，不为单次端口重试或单场景进度单独打包：

1. v1.3 release freeze review；
2. original 3DGS 3K reference smoke review；
3. all-method 3K qualification review；
4. six-scene formal regression review；
5. final paper table/figure provenance package。

最终论文主表：methods x six scenes，报告 H/Z/3D RMSE、coverage、failure rate 和 efficiency。附表提供 per-point residual、Sim(3) conditioning、scatter、seeds 和 diagnostics。所有表格由 machine-readable CSV/JSON 自动生成，人工不得抄写数值。

## 10. 推荐立即执行顺序

1. 实现并冻结 transactional v1.3 release；
2. 实现 evaluator v1.3 loader 与 real-release tests；
3. 冻结 common R8 training/camera/packet contract；
4. 运行 original 3DGS 3K end-to-end smoke；
5. 审核通过后，按 core list 逐个完成 3K qualification；
6. qualification 全部汇总后，再启动六场景正式矩阵；
7. primary 完成后运行 seeds 与 sensitivity；
8. 自动生成最终表格、图和审计包。

GPU 现在不需要用于 F0-F4；从 F5 的 original 3DGS 3K training 开始需要 GPU。批量 GPU 实验不得早于 v1.3 release 与 evaluator freeze。
