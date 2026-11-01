"""Inspect shuffled training batches; no model or optimizer is created."""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from inspect_dinov3_classifier import file_hash, load_verified_cache


def make_train_loader(arrays, batch_size, seed):
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    indices = arrays['train_idx']
    dataset = TensorDataset(torch.from_numpy(arrays['features'][indices]),
                            torch.from_numpy(arrays['labels'][indices]),
                            torch.from_numpy(indices.copy()))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      num_workers=0, generator=generator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/baseline_convnext_tiny.json').read_text(encoding='utf-8'))
    arrays, cache_report, _ = load_verified_cache(root, config)
    loader = make_train_loader(arrays, args.batch_size, args.seed)
    seen, batch_sizes = [], []
    first_shapes = None
    for features, labels, indices in loader:
        if first_shapes is None:
            first_shapes = [list(features.shape), list(labels.shape)]
        if not np.array_equal(features.numpy(), arrays['features'][indices.numpy()]):
            raise RuntimeError('Feature/sample alignment mismatch')
        if not np.array_equal(labels.numpy(), arrays['labels'][indices.numpy()]):
            raise RuntimeError('Label/sample alignment mismatch')
        seen.extend(indices.tolist())
        batch_sizes.append(len(labels))
    if sorted(seen) != sorted(arrays['train_idx'].tolist()):
        raise RuntimeError('Training samples missing, duplicated or incorrectly included')
    if set(seen) & (set(arrays['val_idx'].tolist()) | set(arrays['test_idx'].tolist())):
        raise RuntimeError('Validation or test samples found in training loader')
    print('Training samples:', len(loader.dataset))
    print('Batch size:', args.batch_size)
    print('First batch features:', tuple(first_shapes[0]))
    print('First batch labels:', tuple(first_shapes[1]))
    print('Batches per epoch:', len(loader))
    print('Last batch samples:', batch_sizes[-1])
    print('Every training sample seen exactly once:', True)
    print('Validation/test samples in training batches:', 0)
    print('Optimizer updates:', 0)
    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(timestamp=now.isoformat(), status='training_batch_inspection_no_training',
                  code_commit=git.stdout.strip() if git.returncode == 0 else None,
                  script_sha256=file_hash(Path(__file__)),
                  helper_sha256=file_hash(root / 'scripts/inspect_dinov3_classifier.py'),
                  config=config, cache_sha256=cache_report['cache_sha256'], seed=args.seed,
                  batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0,
                  sample_order=seen, batch_sizes=batch_sizes, sample_count=len(seen),
                  first_batch_shapes=first_shapes, all_training_samples_once=True,
                  validation_test_overlap=0, optimizer_updates=0, torch_version=torch.__version__)
    destination = root / 'logs' / ('dinov3_batches_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
