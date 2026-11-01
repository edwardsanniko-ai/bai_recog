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

## EUPE ConvNeXt

`scripts/vendor/eupe_convnext.py` is an unmodified copy of
`eupe/models/convnext.py` from https://github.com/facebookresearch/EUPE,
commit `f6cdf04d82b97009941a27e9bed3ca17ac7e0366`, retrieved 2026-11-01.
Byte SHA256: `2a2ffa179c6c12dcb1e250818f7b5792906a5edac5374757f92e304f7b312336`.
The full FAIR Noncommercial Research License is retained as `scripts/vendor/EUPE_LICENSE`.
This source remains under its original license, independently of our wrapper scripts.

Our wrapper constructs the same Tiny architecture as the official hub entry point,
loads the pinned official checkpoint, freezes all parameters and uses
`forward_features(...)["x_norm_clstoken"]` for whole-image features.
Preprocessing follows the official 256-square README transform; inference uses float32.
Weights remain local in ignored `hf_cache/` and are not redistributed.
