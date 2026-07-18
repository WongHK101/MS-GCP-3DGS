# Original 3DGS serializer compatibility approval

External GPT review accepted the frozen 5K formal scene result and classified the first 10K attempt as:

```text
training_compute_reached_iteration_30000_but_formal_model_not_materialized
```

The review approved a serializer-only memory-safety compatibility patch based on official original 3DGS commit `2eee0e26d2d5fd00ec462df47752223952f6bf4e`. The allowed source change is limited to the memory implementation of `GaussianModel.save_ply()` plus parity/RSS tests, compatibility provenance, and non-invasive memory diagnostics.

The following remain frozen and unchanged: seed 0, official 30K optimization, graphdeco `--resolution -1`, training images/cameras/SfM, release v1.3.0, formal checkpoint policy, metric packet v2, `alpha_normalized_expected_camera_z=M1/A`, camera-z semantics, patch, aggregation, Sim(3), pointset, and split.

The patch must pass synthetic bitwise parity, immutable 5K real-Ply parity, and independent-process RSS evidence before a from-scratch 10K retry. A successful 10K authorizes continuation through 20K, 50K, and 100K in the same supervised batch. The failed 10K iteration-7000 PLY remains infrastructure evidence only and must not be resumed or promoted to a formal result.

Frozen 5K identities accepted by the review:

- iteration-30000 PLY SHA-256: `88df8eb33ccf22d37381ca7200bac8c7eb8225616f319eaac5fdfd8e538024d0`
- compatibility wrapper SHA-256: `30036a08df6f6f6a06c08921db460aa7aed67d8a43686e22a8bf7c37fd6f6b64`
- checkpoint RMSE-H/Z/3D: `0.14050161023076604 / 0.3568420666110885 / 0.3835061446453116 m`

Review decision:

```text
5K formal scene pipeline: PASS
10K failed run: BLOCKED_AT_FORMAL_MODEL_SERIALIZATION
serializer-only memory-safe fix: APPROVED
higher-memory environment: FALLBACK, NOT CURRENTLY REQUIRED
10K from-scratch retry after parity gates: APPROVED
20K/50K/100K continuation after successful 10K: APPROVED
other methods: NOT INCLUDED
```
