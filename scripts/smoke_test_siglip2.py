"""Load one training image with the public, pinned SigLIP 2 vision model."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root.parent / 'bai_data')
    args = parser.parse_args()
    config = json.loads((root / 'configs/baseline_siglip2.json').read_text(encoding='utf-8'))
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    index = manifest['split_indices']['train'][0]
    record = manifest['records'][index]
    image_path = args.data_root / record['relative_image']
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if digest != record['sha256']:
        raise RuntimeError('Image differs from the audited dataset manifest')
    with Image.open(image_path) as original:
        image = original.convert('RGB')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Model:', config['backbone'], flush=True)
    print('Device:', device, flush=True)
    print('First run downloads the official checkpoint to local hf_cache.', flush=True)
    kwargs = dict(revision=config['revision'], cache_dir=str(root / 'hf_cache'),
                  token=False, trust_remote_code=False)
    processor = AutoImageProcessor.from_pretrained(config['backbone'], **kwargs)
    model = SiglipVisionModel.from_pretrained(config['backbone'], **kwargs).eval().to(device)
    model.requires_grad_(False)
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        features = model(**inputs).pooler_output
    if tuple(features.shape) != (1, 768) or not torch.isfinite(features).all().item():
        raise RuntimeError('Unexpected feature shape or non-finite values')
    print('Input shape:', tuple(inputs['pixel_values'].shape))
    print('Feature shape:', tuple(features.shape))
    print('All finite:', True)
    memory = torch.cuda.memory_allocated(device) / 2**30 if device.type == 'cuda' else None
    if memory is not None:
        print('Allocated GPU GiB:', round(memory, 3))
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=False)
    now = datetime.now(timezone(timedelta(hours=8)))
    report = {
        'timestamp': now.isoformat(), 'status': 'single_image_forward_passed_not_trained',
        'config': config, 'code_commit': git.stdout.strip() if git.returncode == 0 else None,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'sample_index': index, 'image_relative': record['relative_image'], 'image_sha256': digest,
        'device': str(device), 'allocated_gpu_gib': memory,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'input_shape': list(inputs['pixel_values'].shape), 'feature_shape': list(features.shape),
        'all_finite': True, 'processor': processor.to_dict(),
    }
    destination = root / 'logs' / ('siglip2_smoke_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
