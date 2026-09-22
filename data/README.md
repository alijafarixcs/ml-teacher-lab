# Data directory

The lessons primarily use datasets bundled with scikit-learn or generate small synthetic datasets with fixed random seeds. This keeps the course reproducible and usable offline.

- `raw/` is for optional, immutable source downloads.
- `processed/` is for derived data that can be recreated.

Large datasets and generated artifacts are intentionally ignored by Git. Every notebook that may use a remote dataset includes a built-in or synthetic fallback and documents the provenance of the data it actually loads.

