# GS-GCP Stage 0 Freeze

Status: implemented contracts; training remains blocked until every readiness gate passes.

## Purpose

Stage 0 freezes the experiment inputs and instrumentation before any method is
trained. It does not produce benchmark metrics and it does not authorize the
six-scene matrix.

The frozen components are:

1. GS-GCP v1.3.0 release identity and integrity root;
2. original 3DGS `-r -1` resolution semantics (1600-pixel width cap);
3. method admission, publication, source commit/tree, license, and role status;
4. per-method clean source/environment/build/run isolation;
5. an external one-Hz GPU and GNU-time resource probe;
6. a 3K-first qualification gate before any full-scene execution.

## Method roles

- Formal core: 3DGS, 2DGS, PGSR, RaDe-GS, GOF, CityGaussianV2.
- Scalability extension: CityGS-X and MetroGS.
- Conditional: GFSGS and QGS.

QGS is formally published but has no recoverable official public
implementation at the audit date, so it is blocked. CityGS-X has a frozen
official source commit but no license file at that commit; qualification stays
blocked pending license clarification. Listing a method never makes it
full-matrix eligible. Every runnable method needs a frozen recipe and complete
3K qualification first.

## Non-invasive resource probe

`code/gcp/run_with_resource_probe.py` launches the unchanged method command as
an argument vector (never through a shell), records host process statistics via
an isolated, SHA-locked GNU time 1.9 binary, and samples explicitly selected
GPUs once per second using `nvidia-smi`. It records wall time, GPU-hours, peak
VRAM, utilization, estimated energy, and host maximum RSS. It does not touch
the loss, autograd graph, or training source. GNU time is extracted under the
GS-GCP tool namespace without modifying the server's global packages.

Before a GPU child starts, the probe requires three one-second idle samples,
device utilization at most 5%, device memory use at most 1,024 MiB, and no
visible compute process. Peak VRAM is the maximum runtime device-memory use
minus that GPU's idle baseline, not raw whole-device memory. This prevents a
concurrent job from being attributed to the evaluated method.

Every GPU phase must use this wrapper. An absent GPU probe, duplicate output
directory, non-zero child exit, or sampler failure makes the phase fail.
AutoDL-901 is the experiment-execution server and must pass the idle gate
immediately before every GPU child launch. AutoDL-740 is an archive/mirror
server; its GPU availability does not authorize or block a formal run.

## Current hard blockers

At implementation time:

- the v1.3.0 release package exists and its local integrity is verifiable, but
  an external GPT PASS is not recorded;
- the immutable six-scene data mirror has been atomically promoted from
  AutoDL-901 to AutoDL-740: 6,267 files and 64,661,981,667 bytes passed full
  manifest/hash validation and are read-only; 740 is retained as archive
  storage, not as the execution server;
- the same 6,267-file source on AutoDL-901 passed a fresh full-content audit on
  2026-07-18 and is the read-only input for formal experiments;
- the independent local repository is clean, but the GitHub remote still uses
  the retired `MS-GCP-3DGS` repository name and must be renamed before public
  release;
- original 3DGS source and its 6.8 GB environment remain intact on AutoDL-901
  and pass Git, package-lock, and extension-import checks; frozen copies on 740
  are disaster-recovery artifacts only;
- only original 3DGS has a pre-registered 3K recipe. Other methods cannot start
  until their method-specific recipe and environment are frozen.

`validate_gs_gcp_stage0.py --require_training_ready` enforces these blockers.
No launcher may bypass that command.
