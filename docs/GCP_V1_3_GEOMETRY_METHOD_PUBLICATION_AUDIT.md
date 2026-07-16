# GCP v1.3 Geometry Method Publication Audit

Audit date: 2026-07-17

Purpose: enforce the rule that the benchmark candidate set contains only work
formally published in peer-reviewed conference proceedings or a journal.
An arXiv-only record is not sufficient.

## Decision rule

`publication_gate_pass` requires at least one of:

1. a publisher DOI page for a journal/proceedings article; or
2. an official conference proceedings page containing the paper and BibTeX.

An arXiv entry, project page, repository badge, or author acceptance statement
can support identity checks but cannot independently pass the gate.

## Audited candidates

| Method | Publisher/proceedings record | Formal venue | Gate | Experiment status |
|---|---|---|---|---|
| 3DGS | https://doi.org/10.1145/3592433 | ACM TOG 42(4), SIGGRAPH 2023 | PASS | core baseline |
| 2DGS | https://doi.org/10.1145/3641519.3657428 | ACM SIGGRAPH 2024 Conference Papers | PASS | core geometry baseline |
| PGSR | https://doi.org/10.1109/TVCG.2024.3494046 | IEEE Transactions on Visualization and Computer Graphics | PASS | core geometry method |
| RaDe-GS | https://doi.org/10.1145/3789201 | ACM TOG 45(2), 2026 | PASS | core geometry method |
| GOF | https://doi.org/10.1145/3687937 | ACM TOG 43(6), 2024 | PASS | core geometry method |
| CityGaussianV2 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/d218ec74edbfc494fa7d7a253951c603-Abstract-Conference.html | ICLR 2025 | PASS | core large-scene method |
| CityGS-X | https://openaccess.thecvf.com/content/ICCV2025/html/Gao_CityGS-X_A_Scalable_Architecture_for_Efficient_and_Geometrically_Accurate_Large-Scale_ICCV_2025_paper.html | ICCV 2025 | PASS | core large-scene method |
| MetroGS | https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_MetroGS_Efficient_and_Stable_Reconstruction_of_Geometrically_Accurate_High-Fidelity_Large-Scale_CVPR_2026_paper.pdf | CVPR 2026 | PASS | core large-scene method |
| QGS | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Quadratic_Gaussian_Splatting_High_Quality_Surface_Reconstruction_with_Second-order_Geometric_ICCV_2025_paper.html | ICCV 2025 | PASS | conditional 3K feasibility |
| GFSGS | https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Geometry_Field_Splatting_with_Gaussian_Surfels_CVPR_2025_paper.html | CVPR 2025 | PASS | conditional 3K feasibility |

All ten listed methods pass the publication gate. QGS and GFSGS remain
conditional only because their UAV/custom-camera integration and metric-packet
feasibility have not passed the 3K smoke; their publication status is not in
question.

## Exclusion rule for future additions

Before source integration or GPU training, a proposed method must be added to
this table with publisher/proceedings evidence. If only an arXiv paper is
available, set `publication_gate=FAIL_ARXIV_ONLY` and do not allocate an
experiment slot. Recheck only after a formal proceedings/journal record exists.
