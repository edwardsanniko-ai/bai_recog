"""Create a shareable manifest from the existing fixed split.

The manifest contains class labels, split indices and relative identifiers. It
deliberately removes absolute Windows paths and never copies image bytes.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPLIT = ROOT.parent / "bai_code" / "models" / "split.json"
OUTPUT = ROOT / "data" / "splits.json"


def main() -> None:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    paths = source["paths"]
    records = []
    for index, (absolute_path, label) in enumerate(zip(paths, source["labels"])):
        path = Path(absolute_path)
        records.append(
            {
                "index": index,
                "class_id": int(label),
                "class_name": source["categories"][int(label)],
                "relative_image": str(Path(path.parent.name) / path.name),
            }
        )

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
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入: {OUTPUT}")
    print(f"图片记录: {len(records)}")
    print({name: len(indices) for name, indices in payload["split_indices"].items()})


if __name__ == "__main__":
    main()
