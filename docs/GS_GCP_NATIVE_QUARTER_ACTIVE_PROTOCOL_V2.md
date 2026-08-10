# GS-GCP 原生 1/4 公平评测协议 v2

状态：**ACTIVE（评测实现合同）**；**GLOBAL TRAINING HOLD（仅 GOF 一次 3K 方法级运行获授权）**
协议冻结日期：2026-08-09；执行状态更新：2026-08-10
协议 ID：`m3m_gcp_native_quarter_geometry_v2`

本文件替代 v1 的执行合同。v1 的工作树资产已清除，只能从 Git 历史追溯；旧
clean-R4/Pillow 路线及其 checkpoint、结果和资格结论同样不能被新实验继承。

## 1. 权威输入

- 数据根固定为 `M3M-GCP-colmap-native-quarter-v1`。
- 六场景图像和 PINHOLE 相机均为 COLMAP 4.0.4
  `image_undistorter --output_type COLMAP --max_image_size 1414` 的直接输出。
- COLMAP 输出后不得再次 resize、重编码、crop、pad 或 EXIF transpose。
- 像素域固定为 `colmap_4_0_4_image_undistorter_pinhole_max_1414`，坐标采用浮点、
  zero-based pixel centres。
- 所有方法必须使用同一份 RGB、相机、holdout 和初始稀疏几何，不得通过各自的 R4
  实现重建输入。

3K 场景有 94 个相机视图，其中训练 82、holdout 12。这里的 82 是训练视图数，与全六
场景共 82 个正式 GCP 点实例没有关系。

## 2. 正式实例与隔离

源 v1.3.0 有 87 个场景—点实例（48 control、39 checkpoint）。本协议用独立、带
SHA-256 的 overlay 隔离五条记录，不修改源发布：

| 场景 | 点 | 原角色 | 新处置 |
|---|---|---|---|
| 100K | `dxl3` | control | diagnostic only |
| 100K | `dyl2` | checkpoint | diagnostic only |
| 100K | `wy3_1` | checkpoint | diagnostic only |
| 100K | `wy3_2` | control | diagnostic only |
| 20K | `dyl2` | control | diagnostic only |

正式集合固定为 82 个实例：45 control、37 checkpoint。当前没有可信的独立屋顶
checkpoint，因此不得发布“屋顶表面精度排名”。评测器必须按 overlay 的
`point_instance_disposition.csv` 和 `observation_semantics.csv` 执行，不得重新启用
隔离点。

## 3. 跨方法公共正式轨

公共轨衡量统一透明度合成统计下的“渲染支撑期望 camera-z 坐标”，不等同于唯一物理
表面。每个适配器输出原始累计量：

\[
w_i=T_i\alpha_i,\qquad A=\sum_iw_i,\qquad M_1=\sum_iw_i z_i.
\]

人工标注浮点像素 \((u,v)\) 的正式取值为：

\[
D(u,v)=\frac{\operatorname{bilinear}(M_1,u,v)}
{\operatorname{bilinear}(A,u,v)}.
\]

实现必须满足：

1. `A`、`M1` 分别以 float64 做四邻域双线性插值，不能先逐像素除法再插值；
2. 仅当 `A_interp > 1e-6` 时有效，分母不加 epsilon；
3. 任一邻域值非有限或邻域越界即无效，不 padding、clamp 或外推；
4. `camera_z` 必须为有限正值并处于冻结 COLMAP 相机单位；
5. 不得先取整像素，也不得改用窗口中值；
6. 同时报告上下左右 `±0.5 px` 的 camera-z 敏感性，模型单位与经公共 Sim(3) 换算的
   米必须分开。

合成适配至少覆盖单平面、倾斜平面、前后双层、深度边界、低透明度、浮点像素、坐标
往返和“先插值原始矩再除法”。

## 4. 多视聚合与覆盖门槛

视角类别和方位由冻结相机姿态及公共 Sim(3) 决定：off-nadir 不超过 5° 为 `nadir`，
否则为 `oblique`；相机相对测量点的平面方位按 45° 划为 8 个 bin。

每个点先在 `(view_class, azimuth_bin)` 组内求三维几何中位数，再对组代表点求三维几何
中位数。禁止逐坐标中位数。

一个点进入误差统计必须同时满足：

- 有效观测数不少于 `max(4, ceil(0.5 × 该点冻结正式观测数))`；
- 有效 `nadir` 不少于 2；
- 有效 `oblique` 不少于 2；
- 有效斜视观测覆盖至少两个不同的 45° 方位 bin，且其中至少一对 bin 的 8-bin 环形
  距离不小于 2（即不能只是相邻 bin）。

最后一条是离散方位多样性门槛，不声称两条实际观测射线的夹角必然达到 90°。例如落在
bin 边界附近的非相邻 bin，实际夹角仍可能小于 90°。本规则选择的是可复现的冻结离散
判据，而不是无法由 bin 编号保证的连续角度判据。

门槛未通过必须记为覆盖失败，不得以“误差太大”为由删除。

## 5. 唯一公共 Sim(3)

每场景只允许 overlay 中冻结的一套

`target_xyz = scale × rotation @ frozen_colmap_model_xyz + translation`。

它仅由保留的 ground controls 和冻结相机/标注三角化结果拟合。所有方法共享同一变换；
不得用方法输出重新拟合、不得用 checkpoint 配准、不得局部拉伸。control 残差只用于
变换审计，checkpoint 才是独立精度依据。

## 6. 完整性、排名与汇总

场景级排名采用硬完整性门槛：只有该场景的所有正式 checkpoint 都通过点级覆盖门槛，
状态才是 `COMPLETE_RANKED`，并允许计算/发布排名。只要缺少一个正式 checkpoint，就必须
标为 `INCOMPLETE_UNRANKED`：

- 不得从已成功的 checkpoint 子集产生正式场景排名；
- 可以报告子集误差，但必须明确标作诊断量；
- 必须报告 checkpoint 有效数/总数、覆盖率和逐点失败原因。

完整场景至少报告水平、垂直、三维 RMSE，三维 median/P95/max，覆盖率，`±0.5 px`
敏感性，以及训练时间、峰值显存/内存、模型和中间文件大小。跨场景总表使用场景宏平均，
仅纳入 `COMPLETE_RANKED` 场景；不得按图像数或点数加权。OOM、超时和无法产生公共包都
如实记录为不完整结果，不强行跑通。

另设方法原生表面次轨。`z50`、网格参数敏感性和边界局部平面在适配一致性完成前均为
诊断项，不与公共轨合成总分。

## 7. 方法池与输入信息分层

候选池固定为 3DGS、2DGS、PGSR、RaDe-GS、GOF、QGS、CityGaussianV2、CityGS-X、
MetroGS；机器登记为 `configs/m3m_gcp_native_quarter_method_registry_v2.json`。

结果必须标注 `rgb_colmap_only` 或 `rgb_colmap_external_geometry_prior`。CityGS-X、
MetroGS 的外部先验型号、权重 SHA、输入分辨率、命令及成本未冻结前不得资格运行。
CityGS-X 冻结提交缺少明确许可证，内部试验和代码再分发必须区分。

## 8. 3DGS 原生 1/4 资格状态

3DGS 的训练配方固定为
`configs/m3m_gcp_native_quarter_3dgs_3k_recipe_v1.json`：官方 2023 训练源码不改，直接读取
3K `train` 根中的 82 张 COLMAP undistorter JPEG，`--resolution 1`、30K iterations、
seed 0；训练过程看不到 GCP、测量坐标或 holdout 角色标签。

正式评测使用独立的 evaluation-only 源码副本。冻结补丁只增加四个 float32 原始累计量
`A/M1/M2/H`，不修改官方训练源码、checkpoint、alpha cutoff 或提前终止规则；归一化包在
CPU 侧派生。当前状态为：

- 补丁对冻结 3DGS/rasterizer 提交可干净应用，静态校验通过；
- 公共算子 CPU 合成预检、目标 GPU/CUDA 构建及冻结 3K 真实 packet-camera 预检均通过；
- seed 0、30K iterations 的 3K 正式训练和公共评测已完成，场景状态为
  `COMPLETE_RANKED`；正式报告为
  `docs/protocol_evidence/3dgs_native_quarter_formal_3k_seed0_30k_v1.json`；
- 正式 checkpoint RMSE 为 3D `0.0417996071 m`、水平 `0.0276110829 m`、高程
  `0.0313820849 m`；checkpoint PLY SHA-256 为
  `461b48e97f31ee6588b5ba3de52d29ed07b4709134f7a155c95bc7c38dba91ff`；
- 该正式运行完成后已重新锁定：`three_k_training_allowed=false`、`rerun_allowed=false`、
  `full_scene_matrix_eligible=false`。

因此这里的全局 `TRAINING HOLD` 不否定已完成的正式结果，也不阻止后续方法在通过全部门禁后
获得一次方法级授权。当前没有方法处于运行授权状态；实时资格和结果状态只以
`configs/m3m_gcp_native_quarter_method_registry_v2.json` 为准。

### 8.1 2DGS 方法级正式状态

2DGS 仍执行同一个 v2 协议，并未引入新的协议版本。源码固定为官方提交
`335ad612f2e783a4e57b9cbc4d1e167bd599fc98`；3K 配方固定为 seed 0、30K iterations、
`--resolution 1`、`--depth_ratio 0`，直接读取同一冻结 `train` 根的 82 张图像。官方训练源码、
训练 rasterizer 和 checkpoint 均不修改。

公共主轨所需的 `A/M1` 已是官方 2DGS rasterizer 的原生输出；独立 evaluation-only 副本只
补齐 packet v2 诊断字段所需的 `M2/H`。固定提交、配方和补丁已通过本地重放与静态校验，
真实输入 loader 已确认载入 82 个训练相机、0 个测试相机和冻结的 61,302 个初始化点。目标
GPU 官方训练扩展与隔离评测扩展均构建通过；单层/双层合成原始矩一致性通过；资格阶段的
1 次迭代技术 smoke 生成了 66 个可逐包复算的真实 packet，但该 smoke 不属于正式结果。
资格证据为 `docs/protocol_evidence/2dgs_native_quarter_gpu_real_3k_qualification_v1.json`。

随后，唯一获准的 3K seed-0/30K 正式运行已完成：

- 训练耗时 `1264.0515 s`，峰值显存 `31833 MiB`，未发生 OOM；最终 PLY 含
  `1,634,781` 个顶点，SHA-256 为
  `3d13956ad22ede5cae6bb5899f51c358a9d5a2d5f6e853980f705b8966454193`；
- 66 个正式 packet 全部通过身份与复算验证，公共评测覆盖 4/4 checkpoints 与 5/5
  controls，未拟合方法专属 Sim(3)，场景状态为 `COMPLETE_RANKED`；
- checkpoint RMSE 为 3D `0.0339083127 m`、水平 `0.0128695182 m`、高程
  `0.0313711519 m`；正式报告为
  `docs/protocol_evidence/2dgs_native_quarter_formal_3k_seed0_30k_v1.json`；
- 运行完成后已重新锁定：`three_k_training_allowed=false`、`rerun_allowed=false`、
  `full_scene_matrix_eligible=false`。其他方法与六场景矩阵仍保持锁定。

该结果只有一个 seed，可作为本 benchmark 的正式单次结果，但不得用于宣称方法间细小差异
具有统计显著性。本次只是执行状态更新，没有修改 v2 的输入、算子、覆盖、Sim(3) 或排名规则。

### 8.2 GOF 方法级资格状态

GOF 继续执行同一个 v2 协议，不产生新的协议版本。源码固定为官方提交
`5245b20e5d11acd6d1ff5af4b890dc2bedd99693`，许可证与三个 vendored 源码树均已记录。3K 配方
固定为 seed 0、30K iterations、`--resolution 1`、`--kernel_size 0`，直接读取同一冻结 `train`
根；不启用 decoupled appearance、ray jitter、重采样或外部几何先验。官方训练源码、训练
rasterizer 和 checkpoint 均不得修改。

静态源码身份、配方和补丁重放已经通过。独立 evaluation-only rasterizer 使用 GOF 官方
前向中同一组 `T*alpha` 权重增加 `M1/M2/H` 累计量，原生累计 alpha 为 `A`。由于官方相机
射线写为 `(x/fx, y/fy, 1)`，其交点参数 `t` 即 camera-z。官方 rendered-image channel 6
为中位/最大深度，不能替代公共 `M1/A`；GOF 原生 opacity level set 或网格仅是方法族诊断
次轨，`physical_surface_claim=false`。

目标 GPU 上的官方训练 rasterizer、simple-knn 与隔离评测 rasterizer 均已构建通过；真实
loader 确认载入 82 个训练相机、0 个测试相机、1414×1025 图像和 61,302 个初始化点。单层、
双层合成原始矩测试通过。1 次迭代技术 smoke 在 11.61 秒完成，峰值显存 2,873 MiB、无 OOM，
但该模型不属于正式结果。随后 66/66 真实 packet 全部生成并逐包复算通过，方差验证失败像元
为 0；公共评测器通过 4/4 checkpoints 与 5/5 controls，未拟合方法专属 Sim(3)。资格证据为
`docs/protocol_evidence/gof_native_quarter_gpu_real_3k_qualification_v1.json`。

因此当前只授权一次新的 GOF 3K、seed 0、30K iterations 正式运行；禁止 resume、覆盖或完成后
重跑。全局训练、其余方法和六场景矩阵仍锁定。资格阶段的 `COMPLETE_RANKED` 只证明管线在
1-iteration 模型上端到端可执行，不作为 GOF benchmark 精度结果。本节只更新执行状态，没有
修改 v2 的输入、公共算子、覆盖、Sim(3) 或排名语义。

## 9. 解锁顺序

单个方法只有在源码/许可证、原生 1/4 recipe、外部先验、原始矩适配器、合成一致性、
冻结 3K 真实 packet-camera 预检及端到端评测全部通过后，才可单独解锁其 3K 资格实验。
解锁某一方法不自动解锁其他方法或六场景矩阵。
3DGS 和 2DGS 的 3K 正式运行均已完成并重锁；当前方法级训练 allowlist 仅含 GOF 的一次
`gcp_3000_20260602/seed0/30000-iteration` 新运行。

当前实现入口：

- 公共算子：`code/gcp/m3m_native_quarter_protocol.py`
- overlay 生成/验证：`code/gcp/build_m3m_native_quarter_protocol_release.py`、
  `code/gcp/validate_m3m_native_quarter_protocol_release.py`
- 公共评测：`code/gcp/evaluate_m3m_native_quarter_geometry.py`
- 3DGS 配方验证：`code/gcp/validate_m3m_native_quarter_3dgs_recipe.py`
- 3DGS renderer 补丁静态验证：`code/gcp/validate_3dgs_native_quarter_renderer_adapter.py`
- 2DGS 配方/补丁静态验证：`code/gcp/validate_m3m_native_quarter_2dgs_static.py`
- GOF 配方/补丁静态验证：`code/gcp/validate_m3m_native_quarter_gof_static.py`
- 九方法登记验证：`code/gcp/validate_m3m_native_quarter_method_registry.py`
- 正式训练 fail-closed 门禁：`code/gcp/check_m3m_native_quarter_formal_launch.py`

任何实现若回退到整数像素、窗口中值、逐坐标中位数、逐方法 Sim(3)、相邻斜视 bin 即
通过，或在 checkpoint 不完整时发布场景排名，均不属于 v2 协议。
