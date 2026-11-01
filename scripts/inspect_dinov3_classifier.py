"""Verify the selected cache and demonstrate an untrained 768-to-6 linear head."""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_cache(root, config):
    report_path = root / config['feature_cache_report']
    report = json.loads(report_path.read_text(encoding='utf-8'))
    cache_path = root / report['cache_relative']
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if file_hash(cache_path) != report['cache_sha256'] or file_hash(manifest_path) != report['manifest_sha256']:
        raise ValueError('Cache or manifest hash differs from the extraction report')
    with np.load(cache_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for name, expected in [('model_id', config['backbone']), ('model_revision', config['revision']),
                           ('manifest_sha256', report['manifest_sha256'])]:
        if arrays[name].item() != expected:
            raise ValueError(f'Cache metadata mismatch: {name}')
    x, y = arrays['features'], arrays['labels']
    feature_dim = config.get('feature_dim', 768)
    if x.shape != (620, feature_dim) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Invalid features')
    if y.shape != (620,) or y.dtype != np.int64:
        raise ValueError('Invalid label shape or type')
    if not np.array_equal(y, [r['class_id'] for r in manifest['records']]):
        raise ValueError('Labels differ from the original manifest')
    if not np.array_equal(arrays['sample_indices'], np.arange(620)):
        raise ValueError('Unexpected sample order')
    if arrays['categories'].tolist() != manifest['categories']:
        raise ValueError('Category order mismatch')
    counts = {}
    all_indices = []
    for cache_key, split_key, size in [('train_idx', 'train', 434), ('val_idx', 'validation', 93), ('test_idx', 'test', 93)]:
        indices = arrays[cache_key]
        if indices.shape != (size,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Invalid split index type or shape')
        if not np.array_equal(indices, manifest['split_indices'][split_key]):
            raise ValueError('Split differs from the original manifest')
        counts[split_key] = len(indices)
        all_indices.extend(indices.tolist())
    if sorted(all_indices) != list(range(620)):
        raise ValueError('Split overlap or missing samples')
    return arrays, report, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42, help='Demo initialization seed; no training')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/baseline_convnext_tiny.json').read_text(encoding='utf-8'))
    arrays, cache_report, counts = load_verified_cache(root, config)

    # The only model built here is the small classifier; DINOv3 stays in the cache.
    torch.manual_seed(args.seed)
    classifier = nn.Linear(768, 6)
    index = arrays['train_idx'][0]
    sample = torch.from_numpy(arrays['features'][index:index + 1])
    with torch.no_grad():
        scores = classifier(sample)
    if tuple(scores.shape) != (1, 6) or not torch.isfinite(scores).all():
        raise RuntimeError('Invalid classifier output')
    parameter_count = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print('Cache verified:', True)
    print('Split sizes:', counts)
    print('Demo seed:', args.seed)
    print('Classifier:', classifier)
    print('Trainable parameters:', parameter_count)
    print('Input shape:', tuple(sample.shape))
    print('Class score shape:', tuple(scores.shape))
    print('Training performed:', False)

    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(timestamp=now.isoformat(), status='untrained_linear_head_demonstration',
                  code_commit=git.stdout.strip() if git.returncode == 0 else None,
                  script_sha256=file_hash(Path(__file__)), config=config,
                  cache_sha256=cache_report['cache_sha256'], split_counts=counts,
                  seed=args.seed, classifier=str(classifier), parameter_count=parameter_count,
                  sample_index=int(index), input_shape=list(sample.shape), output_shape=list(scores.shape),
                  torch_version=torch.__version__, device='cpu', training_performed=False)
    destination = root / 'logs' / ('dinov3_head_demo_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
