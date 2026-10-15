# Vendored EmbeddingGemma core

Source: https://github.com/ikcore/ai-embeddinggemma-2
Commit: `2369ca8eeebee30785b7a6eff1ab7968e5a56404`
Subdirectory: `crates/embeddinggemma_core/src`
License: MIT; upstream copyright and terms are preserved in `LICENSE`.

Rust source is copied unchanged from this commit, including upstream inline test
modules (no new tests are authored). Upstream examples, integration tests, API
server and MLX dependencies are not included. The standalone Cargo.toml replaces
workspace inheritance, disables default Hub access, omits MLX, and pins both
Candle dependencies to the exact mistral.rs Candle Git revision
`66a8cf184a5a519671454066b1b9efd446ec9f5c`. The upstream workspace manifest is
retained as `UPSTREAM-Cargo.toml` for dependency provenance. Upstream native
safetensors only; model weights are not distributed here and have their own terms.
