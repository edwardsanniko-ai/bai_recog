"""Extract frozen DINOv3 features in audited sample order, without training."""

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
from transformers import AutoImageProcessor, AutoModel


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder(root, config, device):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = dict(revision=config['revision'], cache_dir=str(root / 'hf_cache'),
                  token=False, trust_remote_code=False, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(config['backbone'], use_fast=True, **kwargs)
    model = AutoModel.from_pretrained(config['backbone'], **kwargs).eval().to(device)
    if model.config.model_type != 'dinov3_convnext':
        raise RuntimeError('Expected DINOv3 ConvNeXt architecture')
    model.requires_grad_(False)
    return processor, model


def extract_batch(records, data_root, processor, model, device):
    images = []
    for record in records:
        path = data_root / record['relative_image']
        if sha256(path) != record['sha256']:
            raise RuntimeError(f"Image changed: {record['relative_image']}")
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    inputs = processor(images=images, return_tensors='pt')
    if tuple(inputs['pixel_values'].shape) != (len(records), 3, 224, 224):
        raise RuntimeError('Unexpected preprocessor output dimensions')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model(**inputs).pooler_output.float().cpu().numpy()
    if features.shape != (len(records), 768) or not np.isfinite(features).all():
        raise RuntimeError('Invalid feature dimensions or non-finite output')
    return features


def save_cache(output, arrays):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing an earlier experiment.
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError('Cache readback differs from extracted arrays')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root.parent / 'bai_data')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Images per extraction batch; not a training hyperparameter')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    config = json.loads((root / 'configs/baseline_convnext_tiny.json').read_text(encoding='utf-8'))
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records, split = manifest['records'], manifest['split_indices']
    if len(records) != 620 or [r['index'] for r in records] != list(range(620)):
        raise RuntimeError('Expected 620 records in audited sample order')
    if len({r['relative_image'] for r in records}) != 620:
        raise RuntimeError('Duplicate sample paths')
    flat = [i for key in ['train', 'validation', 'test'] for i in split[key]]
    if sorted(flat) != list(range(620)) or [len(split[k]) for k in ['train', 'validation', 'test']] != [434, 93, 93]:
        raise RuntimeError('Invalid fixed split')
    labels = np.array([r['class_id'] for r in records], dtype=np.int64)
    if len(manifest['categories']) != 6 or labels.min() < 0 or labels.max() >= 6:
        raise RuntimeError('Invalid class labels')
    if any(r['class_name'] != manifest['categories'][r['class_id']] for r in records):
        raise RuntimeError('Class names and IDs disagree')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Model:', config['backbone'], flush=True)
    print('Device:', device, flush=True)
    print('Loading local checkpoint; no download, training, or evaluation.', flush=True)
    processor, model = load_encoder(root, config, device)
    batches = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        batches.append(extract_batch(batch, args.data_root, processor, model, device))
        print(f'Processed: {start + len(batch)}/{len(records)}', flush=True)
    features = np.concatenate(batches, axis=0)
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime('%Y%m%d_%H%M%S_%f')
    output = root / 'cache' / ('dinov3_features_' + run_id + '.npz')
    arrays = dict(features=features, labels=labels, sample_indices=np.arange(620, dtype=np.int64),
                  categories=np.array(manifest['categories']), model_id=np.array(config['backbone']),
                  model_revision=np.array(config['revision']), manifest_sha256=np.array(sha256(manifest_path)),
                  train_idx=np.array(split['train'], dtype=np.int64),
                  val_idx=np.array(split['validation'], dtype=np.int64),
                  test_idx=np.array(split['test'], dtype=np.int64))
    save_cache(output, arrays)
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    weight_path = root / 'hf_cache' / ('models--' + config['backbone'].replace('/', '--')) / 'snapshots' / config['revision'] / 'model.safetensors'
    report = dict(
        timestamp=now.isoformat(), status='features_cached_no_training_or_metrics', config=config,
        code_commit=git.stdout.strip() if git.returncode == 0 else None,
        script_sha256=sha256(Path(__file__)), manifest_sha256=sha256(manifest_path),
        weights_sha256=sha256(weight_path), cache_relative=output.relative_to(root).as_posix(),
        cache_sha256=sha256(output), feature_shape=list(features.shape), label_shape=list(labels.shape),
        dtype=str(features.dtype), all_finite=True, readback_identical=True,
        split_counts={k: len(v) for k, v in split.items()}, extraction_batch_size=args.batch_size,
        all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        processor=processor.to_dict(), use_fast=True, allow_tf32=False,
        torch=torch.__version__, transformers=transformers.__version__, device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        feature_normalization='none; raw pooler_output',
    )
    report_path = root / 'logs' / ('dinov3_cache_' + run_id + '.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Feature shape:', features.shape)
    print('Label shape:', labels.shape)
    print('All finite:', True)
    print('Readback identical:', True)
    print('Saved features:', output)
    print('Saved report:', report_path)


if __name__ == '__main__':
    main()
