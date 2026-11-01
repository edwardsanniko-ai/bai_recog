"""Demonstrate cross-entropy on one training sample; no parameter updates."""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess

import torch
from torch import nn

from inspect_dinov3_classifier import file_hash, load_verified_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42, help='Same demo seed as classifier inspection')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/baseline_convnext_tiny.json').read_text(encoding='utf-8'))
    arrays, cache_report, _ = load_verified_cache(root, config)
    torch.manual_seed(args.seed)
    classifier = nn.Linear(768, 6)
    index = int(arrays['train_idx'][0])
    sample = torch.from_numpy(arrays['features'][index:index + 1])
    target = torch.from_numpy(arrays['labels'][index:index + 1])

    with torch.no_grad():
        scores = classifier(sample)
        # CrossEntropyLoss takes RAW scores, not softmax probabilities.
        loss = nn.CrossEntropyLoss()(scores, target)
        probabilities = scores.softmax(dim=1)
        true_probability = probabilities[0, target.item()]
        # Float64 calculation is for demonstrating -log(p_true), not training.
        manual_loss = -scores.double().softmax(dim=1)[0, target.item()].log()
    torch.testing.assert_close(loss.double(), manual_loss, rtol=1e-5, atol=1e-6)
    if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
        raise RuntimeError('Non-finite loss or probabilities')
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    categories = arrays['categories'].tolist()
    true_name = categories[target.item()]
    prediction = categories[scores.argmax(dim=1).item()]
    print('Demo seed:', args.seed)
    print('True category:', true_name)
    print('Predicted category (untrained):', prediction)
    print('Probabilities by category:')
    for name, probability in zip(categories, probabilities[0].tolist()):
        print(f'  {name}: {probability:.6f}')
    print(f'True category probability: {true_probability.item():.6f}')
    print(f'Cross-entropy loss: {loss.item():.6f}')
    print('Matches -log(p_true):', True)
    print('Training performed:', False)

    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(timestamp=now.isoformat(), status='untrained_single_training_sample_loss_demo',
                  code_commit=git.stdout.strip() if git.returncode == 0 else None,
                  script_sha256=file_hash(Path(__file__)),
                  helper_sha256=file_hash(root / 'scripts/inspect_dinov3_classifier.py'),
                  config=config, cache_sha256=cache_report['cache_sha256'], seed=args.seed,
                  sample_index=index, categories=categories, scores=scores[0].tolist(),
                  probabilities=probabilities[0].tolist(), true_category=true_name,
                  predicted_category=prediction, true_probability=true_probability.item(),
                  cross_entropy=loss.item(), negative_log_probability=manual_loss.item(),
                  loss_formula_matches=True, training_performed=False,
                  torch_version=torch.__version__, device='cpu')
    destination = root / 'logs' / ('dinov3_loss_demo_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
