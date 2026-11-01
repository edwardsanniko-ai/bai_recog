"""Validation-only cosine 1NN evaluation of frozen pretrained DINOv3 features."""

from datetime import datetime, timezone, timedelta
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from inspect_dinov3_classifier import file_hash, load_verified_cache


def cosine_nearest_neighbor(features, reference_indices, query_indices, labels):
    references = np.sort(reference_indices)
    if len(np.unique(references)) != len(references) or np.intersect1d(references, query_indices).size:
        raise ValueError('Duplicate references or reference/query overlap')
    reference_features = features[references].astype(np.float64)
    query_features = features[query_indices].astype(np.float64)
    for data in (reference_features, query_features):
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        if not np.isfinite(data).all() or not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError('Non-finite or zero-length features')
        data /= norms
    similarities = query_features @ reference_features.T
    local_indices = similarities.argmax(axis=1)
    nearest_indices = references[local_indices]
    nearest_scores = similarities[np.arange(len(query_indices)), local_indices]
    return labels[nearest_indices], nearest_indices, nearest_scores


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/baseline_convnext_tiny.json').read_text(encoding='utf-8'))
    protocol = config['evaluation_protocol']
    if config['allow_parameter_updates'] or config['classifier'] != 'cosine_1nn':
        raise ValueError('This script requires frozen cosine 1NN evaluation')
    if protocol['decision_rule'] != 'cosine_nearest_neighbor' or protocol['neighbors'] != 1:
        raise ValueError('Unsupported decision rule')
    arrays, cache_report, _ = load_verified_cache(root, config)
    references, queries = arrays['train_idx'], arrays['val_idx']
    predictions, nearest, similarities = cosine_nearest_neighbor(
        arrays['features'], references, queries, arrays['labels'])
    truth = arrays['labels'][queries]
    categories = arrays['categories'].tolist()
    class_ids = list(range(len(categories)))
    per_class = classification_report(truth, predictions, labels=class_ids,
                                      target_names=categories, output_dict=True, zero_division=0)
    metrics = dict(accuracy=accuracy_score(truth, predictions),
                   macro_precision=per_class['macro avg']['precision'],
                   macro_recall=per_class['macro avg']['recall'],
                   macro_f1=per_class['macro avg']['f1-score'],
                   weighted_f1=per_class['weighted avg']['f1-score'])
    matrix = confusion_matrix(truth, predictions, labels=class_ids)
    if matrix.sum() != 93 or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError('Invalid validation metrics')
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime('%Y%m%d_%H%M%S_%f')
    csv_path = root / 'results' / ('dinov3_1nn_validation_' + run_id + '.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_index', 'true_class_id', 'true_class', 'predicted_class_id',
                         'predicted_class', 'reference_index', 'cosine_similarity', 'correct'])
        for i, actual, predicted, reference, similarity in zip(queries, truth, predictions, nearest, similarities):
            writer.writerow([int(i), int(actual), categories[actual], int(predicted),
                             categories[predicted], int(reference), float(similarity), bool(actual == predicted)])
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    report = dict(timestamp=now.isoformat(), status='frozen_cosine_1nn_validation_evaluated',
                  model_description='DINOv3 ConvNeXt-Tiny frozen pretrained features + cosine 1NN',
                  code_commit=git.stdout.strip() if git.returncode == 0 else None,
                  script_sha256=file_hash(Path(__file__)),
                  helper_sha256=file_hash(root / 'scripts/inspect_dinov3_classifier.py'),
                  config=config, cache_sha256=cache_report['cache_sha256'],
                  reference_split='train', reference_count=len(references), reference_indices=references.tolist(),
                  evaluation_split='validation', evaluation_count=len(queries), query_indices=queries.tolist(),
                  optimizer_updates=0, classifier_trained=False, teaching_weights_used=False,
                  test_evaluated=False, uses_labeled_references=True, zero_shot=False,
                  categories=categories, metrics=metrics, per_class=per_class,
                  confusion_matrix=matrix.tolist(), confusion_matrix_rows='true', confusion_matrix_columns='predicted',
                  predictions_relative=csv_path.relative_to(root).as_posix(), predictions_sha256=file_hash(csv_path),
                  numpy_version=np.__version__, sklearn_version=sklearn.__version__, device='cpu')
    report_path = root / 'logs' / ('dinov3_1nn_validation_' + run_id + '.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Method: frozen DINOv3 + cosine nearest neighbor (k=1)')
    print('Reference images:', len(references))
    print('Validation images:', len(queries))
    for name, value in metrics.items():
        print(f'Validation {name}: {value:.6f}')
    print('Optimizer updates:', 0)
    print('Test evaluated:', False)
    print('Saved predictions:', csv_path)
    print('Saved report:', report_path)


if __name__ == '__main__':
    main()
