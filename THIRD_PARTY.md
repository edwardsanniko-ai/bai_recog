# Third-party code

`scripts/vendor/qwen3_vl_embedding.py` is the Qwen3-VL-Embedding inference helper
from https://github.com/QwenLM/Qwen3-VL-Embedding (Apache License 2.0).
The full license is retained in `scripts/vendor/QWEN_LICENSE`.

It was copied from the pre-existing local `Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py`
on 2026-11-01, with newline normalization only. The local source file's byte SHA256 was
`0e447b6a80e05527f3fd437ab4f1617a6e7d0df102b8164348bbce787a41b267`.
No upstream Git commit was available for this local archive; experiment reports record
the vendored file SHA256 rather than inventing a source revision. The model weights
have their own independently pinned Hugging Face snapshot revision.

Our smoke-test script uses the helper's input template, preprocessing, forward pass,
and last-non-padding-token pooling, then converts to float32 for L2 normalization.
It explicitly freezes parameters and rejects missing image inputs instead of silently
accepting the helper's error fallback to a text-only NULL input. Model weights and
third-party datasets are not distributed in this repository.
