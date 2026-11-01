"""One optimizer update on one training sample; a teaching demo, not a baseline run."""

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import subprocess

import torch
from torch import nn

from inspect_dinov3_classifier import file_hash, load_verified_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', type=float, required=True, help='Learning rate for this single update')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/baseline_convnext_tiny.json').read_text(encoding='utf-8'))
    if not config.get('allow_parameter_updates', False):
        parser.error('Parameter updates are disabled for pretrained-model screening. This historical teaching demo is inactive.')
    arrays, cache_report, _ = load_verified_cache(root, config)
    torch.manual_seed(args.seed)
    classifier = nn.Linear(768, 6)
    initial_state = {key: value.detach().clone() for key, value in classifier.state_dict().items()}
    index = int(arrays['train_idx'][0])
    sample = torch.from_numpy(arrays['features'][index:index + 1])
    target = torch.from_numpy(arrays['labels'][index:index + 1])
    criterion = nn.CrossEntropyLoss()
    # Weight decay is explicitly zero for this isolated learning-rate demonstration.
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.0)

    # Clear any earlier gradients; compute the loss with gradient tracking enabled.
    optimizer.zero_grad(set_to_none=True)
    scores_before = classifier(sample)
    loss_before = criterion(scores_before, target)
    if not torch.isfinite(loss_before):
        raise RuntimeError('Non-finite loss before update')
    # Backpropagation computes gradients; it does not itself change the parameters.
    loss_before.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in classifier.parameters()):
        raise RuntimeError('Missing or non-finite gradients')
    gradient_norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in classifier.parameters()])).item()
    unchanged_after_backward = all(torch.equal(initial_state[k], v) for k, v in classifier.state_dict().items())
    if not unchanged_after_backward:
        raise RuntimeError('Parameters changed before optimizer.step()')
    # This is the ONLY parameter update in the demo.
    optimizer.step()
    with torch.no_grad():
        scores_after = classifier(sample)
        loss_after = criterion(scores_after, target)
        p_before = scores_before.detach().softmax(dim=1)[0, target.item()].item()
        p_after = scores_after.softmax(dim=1)[0, target.item()].item()
    if not torch.isfinite(loss_after):
        raise RuntimeError('Non-finite loss after update')
    updated_state = {key: value.detach().clone() for key, value in classifier.state_dict().items()}
    changed = any(not torch.equal(initial_state[key], value) for key, value in updated_state.items())
    max_change = max((updated_state[k] - initial_state[k]).abs().max().item() for k in initial_state)
    print('Learning rate:', args.lr)
    print(f'Loss before: {loss_before.item():.6f}')
    print(f'Loss after: {loss_after.item():.6f}')
    print(f'True probability before: {p_before:.6f}')
    print(f'True probability after: {p_after:.6f}')
    print('Parameters unchanged after backward:', unchanged_after_backward)
    print('Parameters changed after step:', changed)
    print('Optimizer updates:', 1)
    print('Formal baseline training:', False)

    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime('%Y%m%d_%H%M%S_%f')
    checkpoint = root / 'checkpoints' / 'teaching' / ('dinov3_one_step_' + run_id + '.pt')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open('xb') as stream:
        torch.save(dict(initial_state=initial_state, updated_state=updated_state,
                        optimizer_state=optimizer.state_dict(), seed=args.seed,
                        learning_rate=args.lr, sample_index=index, optimizer_updates=1,
                        cache_sha256=cache_report['cache_sha256'], teaching_only=True), stream)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for state_key, expected in [('initial_state', initial_state), ('updated_state', updated_state)]:
        if any(not torch.equal(saved[state_key][k], v) for k, v in expected.items()):
            raise RuntimeError('Checkpoint readback mismatch')
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(timestamp=now.isoformat(), status='single_training_sample_one_update_teaching_demo',
                  code_commit=git.stdout.strip() if git.returncode == 0 else None,
                  script_sha256=file_hash(Path(__file__)),
                  helper_sha256=file_hash(root / 'scripts/inspect_dinov3_classifier.py'),
                  config=config, cache_sha256=cache_report['cache_sha256'], seed=args.seed,
                  sample_index=index, true_category=arrays['categories'][target.item()].item(),
                  optimizer='AdamW', learning_rate=args.lr, weight_decay=0.0,
                  optimizer_betas=list(optimizer.defaults['betas']), optimizer_eps=optimizer.defaults['eps'],
                  gradient_norm=gradient_norm, max_parameter_change=max_change,
                  loss_before=loss_before.item(), loss_after=loss_after.item(),
                  true_probability_before=p_before, true_probability_after=p_after,
                  unchanged_after_backward=unchanged_after_backward, changed_after_step=changed,
                  optimizer_updates=1, formal_baseline_training=False,
                  checkpoint_relative=checkpoint.relative_to(root).as_posix(),
                  checkpoint_sha256=file_hash(checkpoint), checkpoint_readback_identical=True,
                  torch_version=torch.__version__, device='cpu')
    destination = root / 'logs' / ('dinov3_update_demo_' + run_id + '.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved local checkpoint:', checkpoint)
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
