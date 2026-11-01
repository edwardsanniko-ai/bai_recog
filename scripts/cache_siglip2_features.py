"""Cache frozen SigLIP 2 features in manifest order; no training or evaluation."""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoImageProcessor, SiglipVisionModel


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_batch(records, data_root, processor, model, device):
    images = []
    for record in records:
        path = data_root / record['relative_image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f"Image changed: {record['relative_image']}")
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    inputs = processor(images=images, return_tensors='pt')
    if tuple(inputs['pixel_values'].shape) != (len(records), 3, 224, 224):
        raise RuntimeError('Unexpected preprocessor output dimensions')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        features = model(**inputs).pooler_output.float().cpu().numpy()
    if features.shape != (len(records), 768) or not np.isfinite(features).all():
        raise RuntimeError('Invalid feature shape or non-finite values')
    return features


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root.parent / 'bai_data')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Images per feature extraction batch, not a training parameter')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    config = json.loads((root / 'configs/baseline_siglip2.json').read_text(encoding='utf-8'))
    if config.get('allow_parameter_updates') is not False or not config['backbone_frozen']:
        raise RuntimeError('This screening step requires a frozen model with no parameter updates')
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = manifest['records']
    if len(records) != 620 or [r['index'] for r in records] != list(range(620)):
        raise RuntimeError('Expected exactly 620 records in original sample order')
    if len({r['relative_image'] for r in records}) != 620:
        raise RuntimeError('Duplicate image paths')
    split = manifest['split_indices']
    flattened = [i for key in ['train', 'validation', 'test'] for i in split[key]]
    if sorted(flattened) != list(range(620)):
        raise RuntimeError('Split indices overlap or do not cover the dataset')
    if [len(split[k]) for k in ['train', 'validation', 'test']] != [434, 93, 93]:
        raise RuntimeError('Unexpected split sizes')
    if len(manifest['categories']) != 6 or any(not 0 <= r['class_id'] < 6 for r in records):
        raise RuntimeError('Invalid class IDs')
    if any(r['class_name'] != manifest['categories'][r['class_id']] for r in records):
        raise RuntimeError('Class names and IDs disagree')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Keep float32 extraction consistent across different batch sizes on RTX GPUs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('Model:', config['backbone'], flush=True)
    print('Device:', device, flush=True)
    print('Loading cached checkpoint; no download or login needed.', flush=True)
    kwargs = dict(revision=config['revision'], cache_dir=str(root / 'hf_cache'),
                  token=False, trust_remote_code=False, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(config['backbone'], use_fast=False, **kwargs)
    model = SiglipVisionModel.from_pretrained(config['backbone'], **kwargs).eval().to(device)
    model.requires_grad_(False)
    batches = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        batches.append(extract_batch(batch, args.data_root, processor, model, device))
        print(f'Processed: {start + len(batch)}/{len(records)}', flush=True)
    features = np.concatenate(batches, axis=0)
    labels = np.array([r['class_id'] for r in records], dtype=np.int64)
    if labels.min() < 0 or labels.max() >= len(manifest['categories']):
        raise RuntimeError('Invalid class labels')
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime('%Y%m%d_%H%M%S_%f')
    output = root / 'cache' / ('siglip2_features_' + run_id + '.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(features=features, labels=labels, sample_indices=np.arange(len(records), dtype=np.int64),
                  categories=np.array(manifest['categories']),
                  model_id=np.array(config['backbone']), model_revision=np.array(config['revision']),
                  manifest_sha256=np.array(file_hash(manifest_path)),
                  train_idx=np.array(split['train'], dtype=np.int64),
                  val_idx=np.array(split['validation'], dtype=np.int64),
                  test_idx=np.array(split['test'], dtype=np.int64))
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError('Cache readback differs from extracted arrays')
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=False)
    weight_path = root / 'hf_cache' / ('models--' + config['backbone'].replace('/', '--')) / 'snapshots' / config['revision'] / 'model.safetensors'
    report = dict(
        timestamp=now.isoformat(), status='features_cached_no_training_or_metrics',
        config=config, code_commit=git.stdout.strip() if git.returncode == 0 else None,
        script_sha256=file_hash(Path(__file__)),
        manifest_sha256=file_hash(manifest_path), weights_sha256=file_hash(weight_path),
        cache_relative=output.relative_to(root).as_posix(),
        cache_sha256=file_hash(output),
        feature_shape=list(features.shape), label_shape=list(labels.shape),
        dtype=str(features.dtype), all_finite=True, readback_identical=True,
        split_counts={k: len(v) for k, v in split.items()},
        extraction_batch_size=args.batch_size, backbone_frozen=True,
        all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        optimizer_updates=0, test_evaluated=False,
        processor=processor.to_dict(), use_fast=False, allow_tf32=False,
        torch=torch.__version__, transformers=transformers.__version__, device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        feature_normalization='none; raw pooler_output',
    )
    report_path = root / 'logs' / ('siglip2_cache_' + run_id + '.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Feature shape:', features.shape)
    print('Label shape:', labels.shape)
    print('All finite:', True)
    print('Readback identical:', True)
    print('Optimizer updates:', 0)
    print('Saved features:', output)
    print('Saved report:', report_path)


if __name__ == '__main__':
    main()
