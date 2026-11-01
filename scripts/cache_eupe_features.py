"""Cache frozen EUPE ConvNeXt-Tiny features; no training or classification."""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import torchvision

from smoke_test_eupe import file_hash, load_model, make_transform, image_features


def verified_manifest(root, config):
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records, split = manifest['records'], manifest['split_indices']
    if len(records) != 620 or [r['index'] for r in records] != list(range(620)):
        raise ValueError('Expected 620 records in audited sample order')
    if len({r['relative_image'] for r in records}) != 620:
        raise ValueError('Duplicate image paths')
    if sorted(i for key in ('train', 'validation', 'test') for i in split[key]) != list(range(620)):
        raise ValueError('Overlapping or incomplete splits')
    if [len(split[k]) for k in ('train', 'validation', 'test')] != [434, 93, 93]:
        raise ValueError('Unexpected split sizes')
    categories = manifest['categories']
    if len(categories) != 6 or any(not 0 <= r['class_id'] < 6 for r in records):
        raise ValueError('Invalid class IDs')
    if any(r['class_name'] != categories[r['class_id']] for r in records):
        raise ValueError('Class names and IDs disagree')
    smoke = json.loads((root / config['smoke_report']).read_text(encoding='utf-8'))
    if smoke['manifest_sha256'] != file_hash(manifest_path):
        raise ValueError('Manifest changed since smoke test')
    for key in ('backbone', 'revision', 'weights_sha256', 'vendor_sha256', 'image_size',
                'preprocessing', 'pooling', 'compute_dtype', 'allow_tf32'):
        if smoke['config'][key] != config[key]:
            raise ValueError(f'Configuration differs from successful smoke test: {key}')
    if (smoke['script_sha256'] != file_hash(root / 'scripts/smoke_test_eupe.py')
            or not smoke['all_parameters_frozen'] or not smoke['all_backbone_weights_loaded']
            or not smoke['all_finite'] or smoke['optimizer_updates'] != 0):
        raise ValueError('Smoke test or helper provenance check failed')
    return manifest


def extract_batch(records, data_root, transform, model, device):
    tensors = []
    for record in records:
        path = data_root / record['relative_image']
        if file_hash(path) != record['sha256']:
            raise ValueError(f"Image changed: sample {record['index']}")
        with Image.open(path) as image:
            tensors.append(transform(image.convert('RGB')))
    inputs = torch.stack(tensors).to(device)
    if tuple(inputs.shape) != (len(records), 3, 256, 256):
        raise ValueError('Unexpected preprocessing shape')
    return image_features(model, inputs).numpy()


def save_cache(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError('Cache readback differs from extracted arrays')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root.parent / 'bai_data')
    parser.add_argument('--batch-size', type=int, default=16, help='Extraction batch size; no training')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    config = json.loads((root / 'configs/baseline_eupe_convnext_tiny.json').read_text(encoding='utf-8'))
    manifest = verified_manifest(root, config)
    manifest_hash = file_hash(root / 'data/splits.json')
    records, split = manifest['records'], manifest['split_indices']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Model:', config['backbone'], flush=True)
    print('Device:', device, flush=True)
    print('Loading local checkpoint; no download, training, or classification.', flush=True)
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model, weights_path, excluded = load_model(root, config, device, local_files_only=True)
    transform = make_transform(config['image_size'])
    batches = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        batches.append(extract_batch(batch, args.data_root, transform, model, device))
        print(f'Processed: {start + len(batch)}/{len(records)}', flush=True)
    features = np.concatenate(batches, axis=0)
    if features.shape != (620, 768) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise RuntimeError('Invalid complete feature array')
    frozen = all(not p.requires_grad for p in model.parameters())
    if not frozen or model.training or file_hash(root / 'data/splits.json') != manifest_hash:
        raise RuntimeError('Frozen model or manifest consistency check failed')
    labels = np.array([r['class_id'] for r in records], dtype=np.int64)
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime('%Y%m%d_%H%M%S_%f')
    output = root / 'cache' / ('eupe_features_' + run_id + '.npz')
    arrays = dict(features=features, labels=labels, sample_indices=np.arange(620, dtype=np.int64),
                  categories=np.array(manifest['categories']), model_id=np.array(config['backbone']),
                  model_revision=np.array(config['revision']), manifest_sha256=np.array(manifest_hash),
                  train_idx=np.array(split['train'], dtype=np.int64),
                  val_idx=np.array(split['validation'], dtype=np.int64),
                  test_idx=np.array(split['test'], dtype=np.int64))
    save_cache(output, arrays)
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(
        timestamp=now.isoformat(), status='features_cached_no_training_or_metrics', config=config,
        code_commit=git.stdout.strip() if git.returncode == 0 else None,
        script_sha256=file_hash(Path(__file__)), helper_sha256=file_hash(root / 'scripts/smoke_test_eupe.py'),
        vendor_sha256=file_hash(root / 'scripts/vendor/eupe_convnext.py'),
        smoke_report_sha256=file_hash(root / config['smoke_report']), manifest_sha256=manifest_hash,
        weights_sha256=file_hash(weights_path), cache_relative=output.relative_to(root).as_posix(),
        cache_sha256=file_hash(output), feature_shape=list(features.shape), label_shape=list(labels.shape),
        dtype=str(features.dtype), all_finite=True, readback_identical=True,
        split_counts={k: len(split[k]) for k in ('train', 'validation', 'test')},
        extraction_batch_size=args.batch_size, all_parameters_frozen=frozen, all_backbone_weights_loaded=True,
        excluded_distillation_projector_keys=excluded, optimizer_updates=0, test_evaluated=False,
        classification_performed=False, feature_normalization='none; raw official x_norm_clstoken',
        preprocessing=config['preprocessing'], allow_tf32=False,
        torch_version=torch.__version__, torchvision_version=torchvision.__version__, device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        peak_allocated_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
        elapsed_seconds=round(time.perf_counter() - started, 3))
    report_path = root / 'logs' / ('eupe_cache_' + run_id + '.json')
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Feature shape:', features.shape)
    print('Label shape:', labels.shape)
    print('All finite:', True)
    print('Readback identical:', True)
    print('All parameters frozen:', frozen)
    print('Optimizer updates:', 0)
    print('Test evaluated:', False)
    print('Saved features:', output)
    print('Saved report:', report_path)


if __name__ == '__main__':
    main()
