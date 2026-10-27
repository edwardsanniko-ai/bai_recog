"""Create a shareable manifest from the existing fixed split.

The manifest contains class labels, split indices and relative identifiers. It
deliberately removes absolute Windows paths and never copies image bytes.
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPLIT = ROOT.parent / "bai_code" / "models" / "split.json"
OUTPUT = ROOT / "data" / "splits.json"
DATA_ROOT = ROOT.parent / "bai_data"


def main() -> None:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    paths = source["paths"]
    assert len(paths) == len(source["labels"]) == 620
    split_indices = {
        name: source[key]
        for name, key in [("train", "train_idx"), ("validation", "val_idx"), ("test", "test_idx")]
    }
    flattened = [i for indices in split_indices.values() for i in indices]
    assert len(flattened) == len(set(flattened)) == 620
    assert sorted(flattened) == list(range(620))
    assert [len(v) for v in split_indices.values()] == [434, 93, 93]
    split_of = {i: name for name, indices in split_indices.items() for i in indices}
    folders = ["geometric patterns", "animal patterns", "antique-object patterns",
               "plant patterns", "chinese-character patterns", "combined patterns"]
    records = []
    hashes = defaultdict(list)
    for index, (absolute_path, label) in enumerate(zip(paths, source["labels"])):
        path = PureWindowsPath(absolute_path)
        assert 0 <= label < 6 and path.parent.name == folders[label]
        relative = (Path(path.parent.name) / path.name).as_posix()
        local = DATA_ROOT / relative
        if not local.is_file():
            raise FileNotFoundError(local)
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        hashes[digest].append(index)
        records.append(
            {
                "index": index,
                "class_id": int(label),
                "class_name": source["categories"][int(label)],
                "relative_image": relative,
                "sha256": digest,
            }
        )

    assert len({r["relative_image"] for r in records}) == 620
    cross_split_duplicates = [indices for indices in hashes.values()
                              if len({split_of[i] for i in indices}) > 1]
    payload = {
        "source": "bai_code/models/split.json",
        "categories": source["categories"],
        "records": records,
        "split_indices": {
            "train": [int(i) for i in source["train_idx"]],
            "validation": [int(i) for i in source["val_idx"]],
            "test": [int(i) for i in source["test_idx"]],
        },
        "raw_images_uploaded": False,
        "image_source": "original bai_data images; old split sample assignments retained",
        "source_split_sha256": hashlib.sha256(SOURCE_SPLIT.read_bytes()).hexdigest(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "sample_count": len(records),
        "class_counts": dict(Counter(r["class_name"] for r in records)),
        "split_counts": {name: len(indices) for name, indices in split_indices.items()},
        "all_files_exist": True,
        "all_indices_unique_and_complete": True,
        "cross_split_byte_identical_groups": cross_split_duplicates,
        "limitation": "SHA256 checks identical bytes only, not near duplicates or shared source motifs.",
    }
    (OUTPUT.parent / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入: {OUTPUT}")
    print(f"图片记录: {len(records)}")
    print({name: len(indices) for name, indices in payload["split_indices"].items()})
    print(f"Cross-split byte-identical groups: {len(cross_split_duplicates)}")
    if cross_split_duplicates:
        raise RuntimeError("Resolve duplicate-image split leakage before training; see data/audit.json")


if __name__ == "__main__":
    main()
