"""Cache native frozen Qwen3-VL-Embedding-2B image embeddings one image at a time."""

import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

# The helper pins local-only loading before importing the model libraries.
from smoke_test_qwen2b import file_hash, image_embedding, load_embedder
import numpy as np
from PIL import Image
import torch
import transformers


def extract_record(embedder, record, data_root):
    path = data_root / record['relative_image']
    if file_hash(path) != record['sha256']:
        raise ValueError(f"Image differs from manifest: index {record['index']}")
    with Image.open(path) as original:
        image = original.convert('RGB')
    features, input_info = image_embedding(embedder, image)
    torch.testing.assert_close(features.norm(dim=1), torch.ones(1), rtol=1e-5, atol=1e-6)
    return features.numpy(), dict(sample_index=record['index'], original_size=list(image.size), **input_info)


def make_arrays(features, manifest, config, manifest_hash):
    split = manifest['split_indices']
    return dict(features=features, labels=np.array([r['class_id'] for r in manifest['records']], dtype=np.int64),
                sample_indices=np.arange(len(manifest['records']), dtype=np.int64),
                categories=np.array(manifest['categories']), model_id=np.array(config['backbone']),
                model_revision=np.array(config['revision']), manifest_sha256=np.array(manifest_hash),
                train_idx=np.array(split['train'], dtype=np.int64),
                val_idx=np.array(split['validation'], dtype=np.int64),
                test_idx=np.array(split['test'], dtype=np.int64))


def save_cache(output, arrays):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError('Cache readback mismatch')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root.parent / 'bai_data')
    args = parser.parse_args()
    config = json.loads((root / 'configs/baseline_qwen2b.json').read_text(encoding='utf-8'))
    manifest_path = root / 'data/splits.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records, split = manifest['records'], manifest['split_indices']
    if len(records) != 620 or [r['index'] for r in records] != list(range(620)):
        raise ValueError('Expected 620 records in original sample order')
    if len({r['relative_image'] for r in records}) != 620 or len(manifest['categories']) != 6:
        raise ValueError('Invalid sample paths or categories')
    for record in records:
        label = record['class_id']
        if not isinstance(label, int) or not 0 <= label < 6 or record['class_name'] != manifest['categories'][label]:
            raise ValueError('Invalid label mapping')
    flattened = [i for key in ['train', 'validation', 'test'] for i in split[key]]
    if sorted(flattened) != list(range(620)) or [len(split[k]) for k in ['train', 'validation', 'test']] != [434, 93, 93]:
        raise ValueError('Invalid fixed split')
    print('Model:', config['backbone'], flush=True)
    print('Extraction batch size: 1; no download, training or evaluation.', flush=True)
    started = time.monotonic()
    embedder, model_path = load_embedder(root, config)
    torch.cuda.reset_peak_memory_stats()
    print('Checking checkpoint SHA256...', flush=True)
    weights_hash = file_hash(model_path / 'model.safetensors')
    features, details = [], []
    for record in records:
        try:
            vector, info = extract_record(embedder, record, args.data_root)
        except Exception:
            print(f"Stopped at sample index {record['index']}; no full cache saved. Settings were not changed.", flush=True)
            raise
        features.append(vector)
        details.append(info)
        print(f'Processed: {len(features)}/620', flush=True)
    features = np.concatenate(features, axis=0)
    if features.shape != (620, 2048) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise RuntimeError('Invalid full feature matrix')
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError('Expected unit-norm image embeddings')
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime('%Y%m%d_%H%M%S_%f')
    output = root / 'cache' / ('qwen2b_features_' + run_id + '.npz')
    arrays = make_arrays(features, manifest, config, file_hash(manifest_path))
    save_cache(output, arrays)
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    frozen = all(not p.requires_grad for p in embedder.model.parameters())
    if not frozen:
        raise RuntimeError('Expected all parameters to remain frozen')
    report = dict(timestamp=now.isoformat(), status='features_cached_no_training_or_metrics',
                  config=config, code_commit=git.stdout.strip() if git.returncode == 0 else None,
                  script_sha256=file_hash(Path(__file__)),
                  helper_sha256=file_hash(root / 'scripts/smoke_test_qwen2b.py'),
                  vendor_sha256=file_hash(root / 'scripts/vendor/qwen3_vl_embedding.py'),
                  manifest_sha256=file_hash(manifest_path), weights_sha256=weights_hash,
                  cache_relative=output.relative_to(root).as_posix(), cache_sha256=file_hash(output),
                  feature_shape=list(features.shape), label_shape=list(arrays['labels'].shape),
                  dtype=str(features.dtype), all_finite=True, readback_identical=True,
                  split_counts={k: len(v) for k, v in split.items()}, extraction_batch_size=1,
                  all_parameters_frozen=frozen, optimizer_updates=0, test_evaluated=False,
                  model_config_file_sha256=file_hash(model_path / 'config.json'),
                  processor=embedder.processor.image_processor.to_dict(), per_image_inputs=details,
                  feature_normalization='per_image_l2_float32_no_fitted_statistics',
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  qwen_vl_utils_version=importlib.metadata.version('qwen-vl-utils'),
                  allow_tf32=False, device=str(embedder.model.device), gpu_name=torch.cuda.get_device_name(),
                  peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                  elapsed_seconds=time.monotonic() - started)
    destination = root / 'logs' / ('qwen2b_cache_' + run_id + '.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Feature shape:', features.shape)
    print('Label shape:', arrays['labels'].shape)
    print('All finite:', True)
    print('Readback identical:', True)
    print('All parameters frozen:', frozen)
    print('Optimizer updates:', 0)
    print('Saved features:', output)
    print('Saved report:', destination)


if __name__ == '__main__':
    main()
