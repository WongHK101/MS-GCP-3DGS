# Legacy Wire Identifiers

The public project name is **GS-GCP Benchmark**.

Release v1.2.x, release v1.3.0, metric packet v2, and their review artifacts
were frozen before the public-name correction. Their JSON schema strings,
serialization magic, release IDs, and root digests are immutable protocol data.
Changing those bytes would invalidate previously reviewed hashes.

Therefore:

1. Existing legacy schema strings are accepted only when validating a frozen
   artifact whose hash and release lineage are known.
2. They are not used as display names, repository names, or new run-directory
   namespaces.
3. New non-release protocol records use the `gs_gcp_*` namespace.
4. Frozen release directories are never rewritten solely for branding.
5. A future release may migrate wire identifiers only through a separately
   reviewed release with explicit lineage; no runtime alias may silently change
   serialized content.

This compatibility boundary preserves both a correct public name and exact
reproducibility of already-reviewed data.
