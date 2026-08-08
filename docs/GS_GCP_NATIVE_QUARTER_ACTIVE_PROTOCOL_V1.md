# GS-GCP 原生 1/4 公平评测协议 v1

状态：**SUPERSEDED BY v2（仅保留历史溯源）**；**TRAINING HOLD（训练未放行）**
日期：2026-08-07
协议 ID：`m3m_gcp_native_quarter_geometry_v1`

当前执行合同见 `docs/GS_GCP_NATIVE_QUARTER_ACTIVE_PROTOCOL_V2.md`。v1 的覆盖与排名规则
不再用于新结果。

本文件替代旧 clean-R4/Pillow 评测路线。旧
`GS_GCP_FAIR_EXPERIMENT_PROTOCOL.md` 仍仅是历史材料，不构成本协议的授权来源。

## 1. 权威输入

- 数据根：`M3M-GCP-colmap-native-quarter-v1`。
- 六场景图像和 PINHOLE 相机来自 COLMAP 4.0.4
  `image_undistorter --output_type COLMAP --max_image_size 1414` 的直接输出。
- COLMAP 输出后不得再次 resize、重编码、crop、pad 或 EXIF transpose。
- 像素域：`colmap_4_0_4_image_undistorter_pinhole_max_1414`。
- 坐标：浮点、zero-based pixel centres。
- 所有方法必须使用相同 RGB、相机、RGB holdout 和初始稀疏几何；方法不得以自有 R4
  实现重新生成输入。

## 2. 正式实例与隔离

源 v1.3.0 有 87 个场景—点实例（48 control、39 checkpoint）。本协议通过独立、带
SHA-256 的 overlay 隔离以下五条记录，不修改源发布：

| 场景 | 点 | 原角色 | 新处置 |
|---|---|---|---|
| 100K | `dxl3` | control | diagnostic only |
| 100K | `dyl2` | checkpoint | diagnostic only |
| 100K | `wy3_1` | checkpoint | diagnostic only |
| 100K | `wy3_2` | control | diagnostic only |
| 20K | `dyl2` | control | diagnostic only |

正式集合固定为 82 个实例：45 control、37 checkpoint。前三类屋顶锚点存在约
3.5–4.1 m 的影像锚点—测量坐标不一致；`dxl3` 的物理锚点身份亦不能独立证明，且有约
0.219 m 高程差。当前没有可信的独立屋顶 checkpoint，因此不得从本版本发布“屋顶表面
精度排名”。

机器可读依据为 overlay 中的 `point_instance_disposition.csv` 和
`observation_semantics.csv`。评测器必须拒绝绕过该清单重新启用隔离点。

## 3. 跨方法公共正式轨

公共轨回答的是：在统一透明度合成统计下，方法给出的“渲染支撑期望 camera-z 坐标”有
多准。它不是唯一物理表面，也不得写成“真实表面深度”。

每个适配器输出原始累计量：

\[
w_i=T_i\alpha_i,\quad A=\sum_iw_i,\quad M_1=\sum_iw_i z_i.
\]

人工标注浮点像素 \((u,v)\) 的唯一正式取值是：

\[
D(u,v)=\frac{\operatorname{bilinear}(M_1,u,v)}
{\operatorname{bilinear}(A,u,v)}.
\]

实现约束：

1. `A` 与 `M1` 分别以 float64 运算做四邻域双线性插值，不能先计算像素级 `M1/A`
   再插值。
2. 仅当 `A_interp > 1e-6` 才有效；分母不加 epsilon。
3. 四邻域任一值非有限即无效。
4. 四邻域越界即无效，不 padding、不 clamp、不外推。
5. `camera_z` 必须为有限正值，并以冻结 COLMAP 相机单位输出。
6. 不允许先四舍五入像素，也不允许对称 `7×7` 窗口中值。
7. 每个观测同时报告上下左右 `±0.5 px` 的最大/中位 camera-z 变化；原始值标作
   COLMAP 模型单位，并另用冻结 Sim(3) 尺度换算为米，不能混淆二者。

合成适配必须至少覆盖：单平面、倾斜平面、前后双层、深度边界、低透明度、浮点像素和
坐标往返。单平面与倾斜平面要和精确偏移射线一致；深度边界应暴露混合差异，而不是把
混合值改称物理首表面。

## 4. 多视聚合与覆盖门槛

视角类别和方位由冻结相机姿态、冻结公共 Sim(3) 确定：

- off-nadir 不超过 5° 为 `nadir`，否则为 `oblique`；
- 相机相对测量点的平面方位按 45° 分 8 个 bin。

每个点先在 `(view_class, azimuth_bin)` 组内求三维几何中位数，再对所有组代表点求三维
几何中位数。这样连续帧较多的飞行条带不会仅凭帧数支配结果；禁止逐坐标中位数。

一个点进入误差统计必须同时满足：

- 有效观测数不少于 `max(4, ceil(0.5 × 该点冻结正式观测数))`；
- 有效 `nadir` 不少于 2；
- 有效 `oblique` 不少于 2。

门槛未通过时必须计入覆盖失败，不得按方法误差删除。正式结果始终同时报告 checkpoint
覆盖率。

## 5. 唯一公共 Sim(3)

每个场景只允许 overlay 中的一套

`target_xyz = scale × rotation @ frozen_colmap_model_xyz + translation`。

它仅由该场景保留的 ground controls 和冻结相机/标注三角化结果拟合。所有方法共享该
变换；不得用方法输出重新拟合，不得使用 checkpoint 配准，也不得局部拉伸。

每场景发布：control/checkpoint 名单、三角化残差、Sim(3) 参数与哈希、控制点留一预测
误差，以及留一变换在全部冻结相机中心上的变化。以下两类留一指标必须分开解释：

- omitted-control prediction：省略控制点自身的外推误差；
- camera-centre transform shift：留一变换相对完整变换对全体相机中心造成的变化。

公共变换本身具有厘米到分米级底噪，尤其 50K/100K 高程；论文不得把它当成无误差真值。

## 6. 结果、轨道和汇总

公共正式轨以 checkpoint 为独立准确度依据，control 残差只用于审计。每场景至少报告：

- 水平、垂直、三维误差的 RMSE；
- 三维误差 median、P95、max；
- checkpoint 有效数/总数和覆盖率；
- 观测失败原因与 `±0.5 px` 敏感性；
- 训练时间、峰值显存、峰值内存、模型/中间文件大小。

跨场景总表使用场景宏平均，不按图像数或点数加权。OOM、超时和无法产生公共包都作为
真实结果记录，不强制所有方法跑通所有场景。

另设方法原生表面次轨：2D surfel/平面、QGS 二次曲面、GOF 等值面、官方网格首交等按
预先冻结的表示族适配器分别报告。该轨不与公共轨合成一个分数。`z50`、网格参数敏感性
和边界局部平面在完成逐贡献排序/适配一致性前均为诊断项。

## 7. 九方法池与外部先验

候选池固定为 3DGS、2DGS、PGSR、RaDe-GS、GOF、QGS、CityGaussianV2、CityGS-X、
MetroGS。机器登记为
`configs/m3m_gcp_native_quarter_method_registry_v1.json`。

- QGS 已有官方公开训练实现与许可证，不再按“无官方实现”阻断。
- CityGS-X 的真实场景路线使用 Depth Anything V2；模型、权重 SHA、输入尺寸、命令、
  时间/显存/存储成本未冻结前不得资格运行。其冻结提交缺少明确许可证，内部试验与代码
  再分发必须区分。
- MetroGS 使用 pointmap 稠密初始化，并在官方准备路线中使用 MoGe-2 等几何先验；具体
  Pi3-Align/VGGT 路线、权重与成本未冻结前不得资格运行。

结果必须标注 `rgb_colmap_only` 或 `rgb_colmap_external_geometry_prior`。两类可以在统一
公共轨展示，但必须分层标注，不能宣称信息条件相同。

## 8. 解锁顺序

本协议落地不等于允许 GPU 训练。单个方法只有在以下项目全部通过后，才可单独解锁 3K
资格实验：

1. 官方源码、提交、许可证状态已登记；
2. 原生 1/4 recipe 和输入隔离已冻结；
3. 外部先验（若有）型号、权重 SHA、命令、分辨率和成本已冻结；
4. 适配器输出原始 `A/M1`，并证明 camera-z 单位回到冻结 COLMAP 相机域；
5. 七类以上合成一致性测试通过；
6. 冻结 3K 相机的 packet-camera CPU 预检通过；
7. 公共评测器能在不拟合方法专属 Sim(3) 的情况下完成端到端验收。

已通过门禁的方法可先跑自己的 3K，不需要等待另外八种方法。六场景全矩阵继续锁定，
直到至少一个 3K 资格结果通过复核且扩展规则另行冻结。

## 9. 实现入口

- 公共算子：`code/gcp/m3m_native_quarter_protocol.py`
- overlay 生成：`code/gcp/build_m3m_native_quarter_protocol_release.py`
- 公共评测：`code/gcp/evaluate_m3m_native_quarter_geometry.py`
- 3DGS CPU 预检：`code/gcp/preflight_3dgs_native_quarter_adapter.py`
- 九方法登记验证：`code/gcp/validate_m3m_native_quarter_method_registry.py`

任何实现若回退到整数像素、窗口中值、逐坐标中位数或逐方法 Sim(3)，均不属于本协议。
