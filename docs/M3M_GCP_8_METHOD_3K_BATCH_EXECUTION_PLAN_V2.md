# M3M-GCP 八方法 3K 批次执行方案 V2

状态：`ACTIVE_EXECUTION_PLAN`

批次 ID：`m3m-gcp-3k-eight-method-seed0-20260818`

评测合同：`m3m_gcp_native_quarter_geometry_v2`（保持不变）

方法注册表：`configs/m3m_gcp_native_quarter_method_registry_v3.json`

## 1. 本批次回答的问题

本批次先在 `gcp_3000_20260602` 上一次性打通剩余八种方法的官方源码、
训练、渲染适配和公共 GCP 三维评测。全部方法处理完后再进行一次集中审核，
不在每种方法后等待外部审核。

本批次只验证协议可执行性和方法技术资格，不改变正式评测指标，也不根据
GCP、LiDAR、PSNR 或中间可视化结果选择源码、先验路线、超参数、seed 或
checkpoint。

## 2. 不可变边界

- 输入固定为 `M3M-GCP-colmap-native-quarter-v1`，根摘要为
  `513f8999fe4b110f15bcbecad7932895781cee755ee9ccd7a14ff10298546d75`。
- 场景固定为 `gcp_3000_20260602`；训练集 82 张，held-out 图像 12 张。
- 所有方法仅运行 seed 0；不做多 seed，也不作统计显著性声明。
- 公共主轨仍为按正式渲染权重累积的 `A/M1` expected camera-z；它不被解释为
  方法原生物理表面。
- 公共 Sim(3)、冻结浮点像素射线、覆盖门、异常值规则和聚合规则均沿用
  `m3m_gcp_native_quarter_geometry_v2`。
- GCP、控制点、LiDAR、正射真值及其派生结果不得被训练、先验生成、调参、
  seed 选择或 checkpoint 选择读取。LiDAR 只允许在模型和 recipe 冻结后追加
  held-out 评测。
- 每种方法使用官方 custom-scene 路线和官方默认预算；若上游提供多条路线，
  必须在查看本 benchmark 结果之前按输入类型选择并冻结唯一一条。不同方法的
  迭代数和预处理成本单独报告，不做结果驱动的预算补偿。
- 正式 run 使用全新、不可覆盖的目录；禁止 resume、覆盖或结果驱动重跑。

## 3. 方法池

| 角色 | 方法 | 输入分层 | 3K 动作 |
|---|---|---|---|
| 复用 | 3DGS | RGB + COLMAP | 复用现有 `COMPLETE_RANKED`，不重训 |
| 复用 | 2DGS | RGB + COLMAP | 复用现有 `COMPLETE_RANKED`，不重训 |
| 新执行 1 | PGSR | RGB + COLMAP | 冻结、资格门、训练、评测 |
| 新执行 2 | RaDe-GS | RGB + COLMAP | 冻结、资格门、训练、评测 |
| 新执行 3 | QGS | RGB + COLMAP | 冻结、资格门、训练、评测 |
| 新执行 4 | GSPrior | RGB + COLMAP，场景内部自约束先验 | 冻结、资格门、训练、评测 |
| 新执行 5 | SOF | RGB + COLMAP | 冻结、资格门、训练、评测 |
| 新执行 6 | CityGaussianV2 | RGB + COLMAP | 冻结 3K 单块路线后执行 |
| 新执行 7 | CityGS-X | 预训练单目深度先验 | 冻结 DAv2 路线后执行 |
| 新执行 8 | MetroGS | 预训练点图/深度先验 | 冻结唯一官方路线后执行 |
| 历史保留 | GOF | RGB + COLMAP | `historical_complete_retired`，不重跑、不扩展 |

CityGS-X 可用于内部复现和数值汇报，但因冻结 commit 缺少明确 LICENSE，不得
二次发布其源码、补丁、环境包或模型权重。源码冻结时还确认 GSPrior 的冻结
commit 同样没有 LICENSE；本批次按同一边界仅作内部复现和数值汇报，不再分发
其代码或权重。两者的数值有效性与再分发许可分开判断。

## 4. 输入分层与排名

结果分两层汇报：

1. `rgb_colmap_only`：只使用冻结 RGB、COLMAP 相机和稀疏初始化；
2. `rgb_colmap_external_geometry_prior`：额外使用冻结的公开预训练几何/深度先验。

GSPrior 的 TSDF 由场景自身 RGB/COLMAP 和训练期渲染内部生成，不属于外部
传感器真值；其构建成本仍必须单列。CityGS-X 与 MetroGS 不得伪装成纯 RGB
方法，也不向纯 RGB 方法注入 LiDAR 来“配平”。

## 5. 每种方法的一次性资格门

每种方法按顺序通过以下机器证据后，批次控制器才可签发该方法唯一一次 3K
launch gate；全局训练权限始终为 false。

1. `SOURCE_FROZEN`：官方 URL、commit、tree、submodule commit、LICENSE
   状态和干净工作树已记录；
2. `RECIPE_FROZEN`：官方 custom-scene 命令、预算、分辨率、初始化、先验、
   checkpoint 规则和所有兼容性改动已哈希；
3. `ENVIRONMENT_READY`：Python/PyTorch/CUDA/扩展版本、构建日志和 GPU
   smoke 已记录；
4. `ADAPTER_CONFORMANT`：评测专用 renderer 明确输出公共 A/M1，并证明不改
   训练源码语义和 checkpoint；
5. `SYNTHETIC_PREFLIGHT_PASS`：平面、倾斜面、多层、边界及无效像素测试通过；
6. `REAL_CAMERA_PREFLIGHT_PASS`：冻结 3K 相机上的 packet、ray、pixel-domain、
   hash、重算和公共 evaluator 门通过；
7. `TECHNICALLY_QUALIFIED`：上述证据封闭且训练 truth-deny 检查通过；
8. `FORMAL_RUN_CONSUMED`：唯一 seed-0 正式训练、packet 导出和评测完成或留下
   不可覆盖的失败证据，随后立即重新锁定该方法。

训练失败、构建失败或 OOM 不能被改写为几何分数。结果状态只能是：

- `COMPLETE_RANKED`：正式 checkpoint、全部 packet 和点级覆盖门均通过；
- `INCOMPLETE_UNRANKED`：训练或评测不完整；已算出的子集数值仅作诊断；
- `TECHNICAL_FAILURE_UNRANKED`：未能形成可评测 checkpoint 或适配器；
- `NOT_ATTEMPTED`：仅用于尚未轮到的方法。

`TECHNICALLY_QUALIFIED` 与结果完整性分开记录。一个方法可在技术上具备六场景
尝试资格，但在某场景因 OOM 成为 `INCOMPLETE_UNRANKED`；失败本身是正式结果，
不要求强行跑通。

## 6. 执行顺序与继续策略

正式训练顺序固定为：

`PGSR -> RaDe-GS -> QGS -> GSPrior -> SOF -> CityGaussianV2 -> CityGS-X -> MetroGS`

可以并行下载源码、构建互相隔离的环境和准备适配器，但 901 上一次只运行一个
正式训练。某方法的非系统性构建失败、方法特有 OOM、单方法高误差或
`INCOMPLETE_UNRANKED` 被完整记录后继续下一方法。

只有下列红线暂停整个批次：冻结输入/相机/SfM 不一致；truth 泄漏；必须改变
正式 A/M1、Sim(3)、覆盖、聚合或 split；需要根据结果改 recipe；他人 GPU
冲突；超出已授权磁盘；原始或冻结结果将被覆盖；或同一系统性管线错误在多个
方法重复出现。

## 7. 存储与清理

- 正式根目录固定为
  `/root/autodl-tmp/runs/m3m-gcp-native-quarter/<method>/gcp_3000_20260602/<run-id>`。
- 环境、源码、构建、先验缓存和运行结果分目录保存，禁止跨方法写入同一源码树。
- 旧 R4、旧协议、废弃 draft、失败的可重建 build cache 不得作为正式输入；
  新入口生效后移除旧的可执行指针，历史证据只通过 Git 历史或明确的
  `historical_*` 状态保留。
- 批次结束前保留所有正式 checkpoint、packet manifest、日志、资源记录和失败
  证据。大模型可暂存 901 持久盘；释放、清盘或删除实例前必须重新决定备份并
  校验 SHA-256。

## 8. 集中审核交付物

八方法均尝试后一次性提交：源码/许可证清单、recipe 与环境哈希、adapter
conformance、每个方法的资格门、GPU/CPU/磁盘成本、训练/评测日志、checkpoint
和 packet 哈希、完整排名表、未排名失败表、纯 RGB 与外部先验分层表，以及
下一阶段 10 方法 × 6 场景的条件执行矩阵。

本文件本身不授权跳过方法门，也不授权六场景扩展。六场景只在本批次集中审核
后按 `TECHNICALLY_QUALIFIED` 方法逐场尝试。
