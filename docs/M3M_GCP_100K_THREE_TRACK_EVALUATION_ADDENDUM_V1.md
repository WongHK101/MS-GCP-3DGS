# M3M-GCP 100K RGB/GCP/LiDAR evaluation addendum v1

Status: review candidate. It is not executable until task
`019ff12c-cb29-7cb2-8fb6-1d82c5f8c54b` returns the exact verdict
`PASS_100K_THREE_TRACK_EVALUATION_ADDENDUM_V1` for the clean addendum
commit/tree and the post-attempt candidate manifest is activated.

## Scope

This addendum changes no training attempt, method budget, checkpoint, seed,
input split, renderer math, GCP metric, LiDAR metric, RGB metric, common Sim(3),
or retry decision. It bridges the already frozen 100K ten-method attempt set to
the project's three orthogonal reporting tracks:

- heldout RGB: PSNR, SSIM and LPIPS-VGG over all 314 frozen test-role images;
- GCP: the existing `m3m_gcp_native_quarter_geometry_v2` point evaluator and
  independent verifier;
- LiDAR: the existing `m3m_gcp_lidar_rendered_surface_v1` evaluator, verifier
  and lightweight archive.

Only methods frozen as `READY_FOR_EVALUATION` receive formal outputs. Failed or
OOM attempts retain null metric paths and are never replaced with zeros.

## Activation boundary

Training and the ten-method attempt freeze remain governed by activation v3 at
commit `e33368db9333f826a3e808ff00c437c1a6c63b82`. The addendum uses a separate
clean checkout and cannot rewrite that base checkout. After all training attempts
finish, the candidate builder binds the immutable base activation, methods
manifest, scene-attempt freeze, model identities, exact 100K formal input,
heldout camera root, 100K RGB registry, and legacy 3DGS GCP adoption receipt.

Formal packet export, GCP evaluation, RGB rendering or LiDAR evaluation is
forbidden until the exact candidate receives the required verdict and an
exclusive read-only activation manifest is created. All formal output roots must
be absent at activation.

## RGB extension

`configs/m3m_gcp_native_quarter_rgb_quality_100k_v1.json` preserves the frozen
RGB metric, quantization, prediction, appearance and aggregation semantics and
adds only the exact 100K scene binding: 2,510 full views, 2,196 train views,
314 test views, and 1,414 x 1,024 pixels.

The RGB camera-root materializer creates an evaluation-only 314-camera loader
root from the frozen test role. Cameras, poses, image bytes and `points3D.ply`
are linked without transformation; an eight-byte zero-point `points3D.bin` is
added only for deterministic COLMAP-loader compatibility. Graphdeco-style
loaders may decode heldout images while constructing camera objects, but the
frozen adapter must clear `original_image` before the renderer is called. No
heldout pixel may select, fit or optimize a parameter.

RGB rendering is independent of metric-depth packets and therefore does not
block packet deletion. It must still bind the same scene-attempt freeze and
three-track activation.

## GCP legacy adoption and new results

The reused 3DGS model already has a complete, independently verified 100K GCP
result. Its 211 packet views are exactly the annotated GCP evaluation subset,
not the later all-2,196-view LiDAR packet set. The candidate builder creates an
immutable adoption receipt that binds the same model SHA, base attempt identity,
formal input, GCP protocol, common Sim(3), old packet manifest, unchanged result
bytes, independent verifier, and proof that all 211 names belong to the frozen
formal split with the exact role counts 187 train and 24 test. Metrics are not
recomputed or rewritten.

Every other READY method first exports a dedicated exact-211-camera GCP packet:
187 frozen train-role cameras plus 24 frozen test-role cameras. Test-role use is
strictly post-freeze geometry evaluation: only frozen poses/intrinsics,
placeholder carrier images, and external GCP observations are present. No test
RGB pixel can reach a prior, training, tuning, checkpoint, seed, or retry
decision. A per-method GCP authorization binds the packet-manifest SHA, model
identity, scene-attempt freeze, fresh output root, frozen GCP evaluator Python
environment, and three-track activation.

## Rolling lifecycle and deletion gate

For each READY method:

1. except for legacy 3DGS adoption, acquire the addendum-wide atomic raw-packet
   mutex and export the dedicated exact-211-camera GCP packet;
2. run the GCP evaluator and independent verifier, build the exact lightweight
   GCP archive, byte-reverify it, then delete the GCP packet, its track state,
   and finally the global mutex; a no-retry failure uses the separately verified
   immutable failure-cleanup receipt instead;
3. only after the GCP lifecycle has released the global mutex, acquire that same
   mutex for a separate exact-2,196-train-view LiDAR packet; the GCP and LiDAR
   raw packets may never coexist;
4. run the unchanged LiDAR launch gate, evaluator and independent verifier,
   build and byte-reverify the exact LiDAR lightweight archive, then delete the
   LiDAR packet, its track state, and finally the global mutex; a no-retry failure
   follows the same fail-closed cleanup contract.

The 2,510-view all-image SfM remains the upstream camera solution, but no
2,510-view raw metric-depth packet is authorized. GCP and LiDAR have separate
packet and archive lifecycles and are released independently.

RGB may run before or after this lifecycle, but its model, 314-view coverage and
output root are frozen in the candidate registry. No RGB, GCP or LiDAR result may
trigger a retry, checkpoint change, hyperparameter change or method substitution.

Final models and bidirectional LiDAR distance arrays remain on 901. Only metrics,
manifests, compact logs, GCP tables and lightweight evidence are pulled locally.
