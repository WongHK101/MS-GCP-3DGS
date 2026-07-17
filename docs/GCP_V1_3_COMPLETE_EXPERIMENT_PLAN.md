# GS-GCP v1.3 实验执行索引

本文件保留原路径以兼容既有审查链接。当前完整、公平且可执行的实验协议已统一到：

[`GS_GCP_FAIR_EXPERIMENT_PROTOCOL.md`](GS_GCP_FAIR_EXPERIMENT_PROTOCOL.md)

关键执行顺序为：冻结 release 与共同输入 -> 每个方法完成 3K 全流水线 -> 通过的
方法扩展到其余五场景 -> 独立复算与统一汇总。正式分辨率为原版 3DGS `-r -1`
对应的 1600-pixel width cap，不再使用历史 R8 recipe。
