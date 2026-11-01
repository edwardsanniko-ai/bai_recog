"""Single-image native Qwen3-VL-Embedding-2B check, without training or classification."""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from PIL import Image
import torch
import torch.nn.functional as F
import transformers

from vendor.qwen3_vl_embedding import Qwen3VLEmbedder


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedder(root, config):
    if config['allow_parameter_updates'] or not config['backbone_frozen']:
        raise ValueError('This step requires frozen inference')
    model_path = (root / config['local_model_path']).resolve()
    if not (model_path / 'model.safetensors').is_file():
        raise FileNotFoundError('Expected local Qwen embedding checkpoint is missing')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This recorded 2B protocol requires a CUDA GPU with bfloat16 support')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    embedder = Qwen3VLEmbedder(
        model_name_or_path=str(model_path), dtype=torch.bfloat16,
        attn_implementation=config['attention_implementation'], local_files_only=True,
        default_instruction=config['instruction'], min_pixels=config['min_pixels'],
        max_pixels=config['max_pixels'], max_length=config['max_length'])
    embedder.model.eval()
    embedder.model.requires_grad_(False)
    embedder.model.config.use_cache = False
    return embedder, model_path


def image_embedding(embedder, image):
    conversation = embedder.format_model_input(image=image)
    inputs = embedder._preprocess_inputs([conversation])
    if 'pixel_values' not in inputs or 'image_grid_thw' not in inputs:
        raise RuntimeError('Image processing failed; refusing a text-only fallback')
    image_tokens = int((inputs['input_ids'] == embedder.model.config.image_token_id).sum().item())
    if image_tokens <= 0:
        raise RuntimeError('No image tokens in input')
    input_info = dict(sequence_length=int(inputs['input_ids'].shape[1]), image_tokens=image_tokens,
                      image_grid_thw=inputs['image_grid_thw'].tolist(),
                      pixel_values_shape=list(inputs['pixel_values'].shape))
    inputs = {key: value.to(embedder.model.device) for key, value in inputs.items()}
    inputs['use_cache'] = False
    with torch.inference_mode():
        outputs = embedder.forward(inputs)
        pooled = embedder._pooling_last(outputs['last_hidden_state'], outputs['attention_mask']).float()
        if tuple(pooled.shape) != (1, 2048) or not torch.isfinite(pooled).all() or pooled.norm().item() == 0:
            raise RuntimeError('Invalid native image embedding')
        features = F.normalize(pooled, p=2, dim=-1)
    return features.cpu(), input_info


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root.parent / 'bai_data')
    args = parser.parse_args()
    config = json.loads((root / 'configs/baseline_qwen2b.json').read_text(encoding='utf-8'))
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    index = manifest['split_indices']['train'][0]
    record = manifest['records'][index]
    image_path = args.data_root / record['relative_image']
    if file_hash(image_path) != record['sha256']:
        raise ValueError('Image differs from audited manifest')
    with Image.open(image_path) as original:
        image = original.convert('RGB')
    print('Model:', config['backbone'], flush=True)
    print('Loading local checkpoint; no download or training.', flush=True)
    torch.cuda.reset_peak_memory_stats()
    embedder, model_path = load_embedder(root, config)
    features, input_info = image_embedding(embedder, image)
    torch.testing.assert_close(features.norm(dim=1), torch.ones(1), rtol=1e-5, atol=1e-6)
    frozen = all(not p.requires_grad for p in embedder.model.parameters())
    print('Feature shape:', tuple(features.shape))
    print('Feature dtype:', str(features.numpy().dtype))
    print('All finite:', bool(torch.isfinite(features).all()))
    print('All parameters frozen:', frozen)
    print('Optimizer updates:', 0)
    print('Allocated GPU GiB:', round(torch.cuda.memory_allocated() / 2**30, 3))
    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(timestamp=now.isoformat(), status='native_image_embedding_smoke_passed_no_classification',
                  code_commit=git.stdout.strip() if git.returncode == 0 else None, config=config,
                  script_sha256=file_hash(Path(__file__)),
                  vendor_sha256=file_hash(root / 'scripts/vendor/qwen3_vl_embedding.py'),
                  manifest_sha256=file_hash(manifest_path), image_sha256=record['sha256'],
                  sample_index=index, image_relative=record['relative_image'], original_size=list(image.size),
                  model_config=embedder.model.config.to_dict(),
                  model_config_file_sha256=file_hash(model_path / 'config.json'),
                  feature_shape=list(features.shape), feature_dtype='float32', all_finite=True,
                  embedding_norm=features.norm().item(), all_parameters_frozen=frozen,
                  optimizer_updates=0, test_evaluated=False, device=str(embedder.model.device),
                  gpu_name=torch.cuda.get_device_name(), peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  qwen_vl_utils_version=importlib.metadata.version('qwen-vl-utils'),
                  processor=embedder.processor.image_processor.to_dict(), input_info=input_info)
    destination = root / 'logs' / ('qwen2b_smoke_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
