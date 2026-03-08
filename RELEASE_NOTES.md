# Release Notes

## v1.0.0 - 2026-03-09

### Highlights
- Prepared repository for team collaboration and clean cloning from GitHub.
- Added full project assets required for reproducible ML workflows.
- Included complete Nepali classification pipeline with trained model artifacts.
- Improved documentation with setup, training, inference, and integration guidance.

### Added
- New `nepali/` pipeline:
  - Data generation and merge scripts
  - Training and prediction scripts
  - Saved checkpoints in `nepali/results/`
  - Best model in `nepali/best_model/`
  - Evaluation outputs in `nepali/analysis/`
- New supporting dataset/artifact files in `civicconnect/data/` and `civicconnect/results/`
- Root-level project documentation improvements in `README.md`

### Changed
- Updated `.gitignore` to keep important project assets tracked while excluding local/dev-only files.
- Removed tracked macOS metadata files (`.DS_Store`) from git tracking.
- Updated model/evaluation artifacts in English pipeline outputs.

### Infrastructure
- Git LFS objects uploaded and synced for large model files.
- Repository now pushes cleanly and is clone-ready for frontend/backend extension work.

### Notes For Contributors
- Run `git lfs install` and `git lfs pull` after cloning.
- Backend can wrap prediction scripts into API endpoints (`/predict/en`, `/predict/ne`).
- Frontend can consume `label` + `confidence` responses from backend APIs.
