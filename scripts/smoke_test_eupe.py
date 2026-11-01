"""One training-image check of frozen official EUPE ConvNeXt-Tiny features."""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2

from vendor.eupe_convnext import ConvNeXt, convnext_sizes


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_transform(size=256):
    # Preserve order and tensor resize semantics from the official EUPE README.
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_model(root, config, device, local_files_only=False):
    if config['allow_parameter_updates'] is not False or not config['backbone_frozen']:
        raise ValueError('This step requires frozen inference')
    if file_hash(root / 'scripts/vendor/eupe_convnext.py') != config['vendor_sha256']:
        raise ValueError('Vendored official source differs from the pinned version')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = Path(hf_hub_download(
        repo_id=config['backbone'], filename=config['weights_filename'],
        revision=config['revision'], cache_dir=str(root / 'hf_cache'),
        token=False, local_files_only=local_files_only))
    if file_hash(path) != config['weights_sha256']:
        raise ValueError('Checkpoint SHA256 differs from official HF LFS metadata')
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = ConvNeXt(**convnext_sizes['tiny'])
    expected = set(model.state_dict())
    missing = sorted(expected - set(state))
    extra = sorted(set(state) - expected)
    # Official hub loads strict=False. Require every backbone key, and only
    # exclude the eight known distillation-projector tensors in this checkpoint.
    allowed_extra = sorted(
        f'projectors.heads.dinov3radio_vitg16.{kind}.layers.{suffix}'
        for kind in ('cls', 'patch')
        for suffix in ('0.weight', '1.weight', '1.bias', '3.weight'))
    if missing or extra != allowed_extra:
        raise ValueError(f'Unexpected checkpoint keys: missing={missing}, extra={extra}')
    for suffix in ('weight', 'bias'):
        if not torch.equal(state['norm.' + suffix], state['norms.3.' + suffix]):
            raise ValueError('Shared LayerNorm checkpoint aliases disagree')
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    return model, path, extra


def image_features(model, inputs):
    with torch.inference_mode():
        features = model.forward_features(inputs)['x_norm_clstoken']
    if features.ndim != 2 or features.shape != (len(inputs), 768):
        raise RuntimeError('Expected EUPE whole-image features [batch,768]')
    if not torch.isfinite(features).all() or torch.any(features.norm(dim=1) == 0):
        raise RuntimeError('Invalid or zero-length features')
    return features.detach().float().cpu()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root.parent / 'bai_data')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((root / 'configs/baseline_eupe_convnext_tiny.json').read_text(encoding='utf-8'))
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    index = manifest['split_indices']['train'][0]
    record = manifest['records'][index]
    image_path = args.data_root / record['relative_image']
    if file_hash(image_path) != record['sha256']:
        raise ValueError('Image differs from audited manifest')
    with Image.open(image_path) as original:
        image = original.convert('RGB')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Model:', config['backbone'], flush=True)
    print('Device:', device, flush=True)
    print('Uses pinned public weights in local hf_cache; downloads only if needed.', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model, weights_path, excluded = load_model(root, config, device, args.local_files_only)
    inputs = make_transform(config['image_size'])(image).unsqueeze(0).to(device)
    features = image_features(model, inputs)
    frozen = all(not p.requires_grad for p in model.parameters())
    assert frozen and not model.training
    print('All backbone weights loaded:', True)
    print('Excluded distillation projector tensors:', len(excluded))
    print('Input shape:', tuple(inputs.shape))
    print('Feature shape:', tuple(features.shape))
    print('Feature dtype:', str(features.numpy().dtype))
    print('All finite:', True)
    print('All parameters frozen:', frozen)
    print('Optimizer updates:', 0)
    print('Test evaluated:', False)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None
    if peak is not None:
        print('Peak allocated GPU GiB:', round(peak, 3))
    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(
        timestamp=now.isoformat(), status='single_image_smoke_passed_no_classification', config=config,
        code_commit=git.stdout.strip() if git.returncode == 0 else None,
        script_sha256=file_hash(Path(__file__)), vendor_sha256=config['vendor_sha256'],
        weights_sha256=file_hash(weights_path), manifest_sha256=file_hash(manifest_path),
        sample_index=index, image_relative=record['relative_image'], image_sha256=record['sha256'],
        original_size=list(image.size), input_shape=list(inputs.shape), feature_shape=list(features.shape),
        feature_dtype='float32', all_finite=True, all_backbone_weights_loaded=True,
        excluded_distillation_projector_keys=excluded, all_parameters_frozen=frozen,
        parameter_count=sum(p.numel() for p in model.parameters()), optimizer_updates=0,
        test_evaluated=False, classification_performed=False, device=str(device),
        torch_version=torch.__version__, torchvision_version=torchvision.__version__, allow_tf32=False,
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        peak_allocated_gpu_gib=peak)
    destination = root / 'logs' / ('eupe_smoke_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
