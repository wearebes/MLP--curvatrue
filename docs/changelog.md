# Neural Curvature Pipeline Changelog

## [2026-03-25] - Architecture Consolidation & Decoupling

### Added
- Created `generate/pde_utils.py` consolidating `reinitializer.py` and `field_builder.py`.
- Created `test/evaluator.py` consolidating the test evaluation flow scripts.
- Designed `tools/verify_pipeline.sh` for an end-to-end integration test validating generations, configs, training, and testing.

### Changed
- Refactored `train/` by merging `model_structure.py` and `inference.py` into a unified `train/model.py`.
- Rewrote the testing pipeline (`test/__main__.py`) and `test/utils.py` to route all result metrics automatically inside the model's `evals/` subfolder.
- Modified dataset evaluations to properly target the generated split source folder using the `--source` parameter.

### Removed
- Removed the old disparate evaluation and generation root-level utilities in favor of namespace packages.
- Deleted broken, disjointed tests such as `evaluate_origin.py` to simplify the test suite. 
