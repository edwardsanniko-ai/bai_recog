"""Create a local visual audit of DINOv3, ResNet50, and ConvNeXt-T errors."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

_DLL_HANDLE = None
if os.name == "nt":
    dll_dir = Path(sys.prefix) / "Library" / "bin"
    if dll_dir.is_dir():
        os.environ["PATH"] = str(dll_dir) + os.pathsep + str(Path(sys.prefix)) + os.pathsep + os.environ["PATH"]
        _DLL_HANDLE = os.add_dll_directory(str(dll_dir))
        modules = list((Path(sys.prefix) / "Lib" / "site-packages" / "PIL").glob("_imaging.cp*-win_amd64.pyd"))
        if len(modules) == 1:
            ctypes.WinDLL(str(modules[0]))

import argparse
from collections import Counter
import csv
import hashlib
import json

from PIL import Image, ImageDraw, ImageFont, ImageOps


MODELS = {
    "D": ("DINOv3", "results/dinov3_1nn_validation_20261101_122548_827845.csv"),
    "R": ("ResNet50", "results/control_resnet50_1nn_validation_20261106_160802_791716.csv"),
    "C": ("ConvNeXt-T", "results/control_convnext_tiny_1nn_validation_20261106_160802_886467.csv"),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def text_lines(draw, xy, lines, font, fill="black", spacing=3):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + spacing


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "top3_error_audit")
    args = parser.parse_args()
    manifest_path = root / "data" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    rows = {}
    source_hashes = {}
    for key, (_, relative) in MODELS.items():
        path = root / relative
        source_hashes[key] = file_hash(path)
        with path.open(encoding="utf-8", newline="") as stream:
            values = list(csv.DictReader(stream))
        rows[key] = {int(value["sample_index"]): value for value in values}
    query_ids = list(rows["D"])
    if any(list(rows[key]) != query_ids for key in rows):
        raise ValueError("Prediction CSV query orders differ")
    if query_ids != manifest["split_indices"]["validation"]:
        raise ValueError("Prediction queries differ from the fixed validation split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patterns = Counter()
    audit = []
    for sample_index in query_ids:
        pattern = "".join("1" if rows[key][sample_index]["correct"] == "True" else "0" for key in "DRC")
        patterns[pattern] += 1
        if pattern == "111":
            continue
        record = records[sample_index]
        item = {
            "sample_index": sample_index,
            "relative_image": record["relative_image"],
            "true_class": record["class_name"],
            "correct_pattern_DRC": pattern,
        }
        for key, (name, _) in MODELS.items():
            row = rows[key][sample_index]
            reference_index = int(row["reference_index"])
            item.update({
                f"{name}_correct": row["correct"],
                f"{name}_predicted": row["predicted_class"],
                f"{name}_reference_index": reference_index,
                f"{name}_reference_image": records[reference_index]["relative_image"],
                f"{name}_cosine_similarity": row["cosine_similarity"],
            })
        audit.append(item)
    csv_path = args.output_dir / "error_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)

    image_size = (210, 150)
    cell_width, row_height = 240, 225
    margin, header_height = 18, 72
    title_font, body_font, small_font = load_font(20, True), load_font(14, True), load_font(12)
    for pattern in sorted({item["correct_pattern_DRC"] for item in audit}):
        selected = [item for item in audit if item["correct_pattern_DRC"] == pattern]
        sheet = Image.new("RGB", (margin * 2 + cell_width * 4, header_height + row_height * len(selected) + margin), "#f5f5f5")
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, 12), f"Error pattern D/R/C={pattern} | 1=correct, 0=wrong | n={len(selected)}", font=title_font, fill="black")
        headings = ["Validation query", "DINOv3 nearest train", "ResNet50 nearest train", "ConvNeXt-T nearest train"]
        for col, heading in enumerate(headings):
            draw.text((margin + col * cell_width, 43), heading, font=small_font, fill="#333333")
        for row_number, item in enumerate(selected):
            top = header_height + row_number * row_height
            sample_index = item["sample_index"]
            query_path = args.data_root / records[sample_index]["relative_image"]
            if file_hash(query_path) != records[sample_index]["sha256"]:
                raise ValueError(f"Image differs from manifest: {sample_index}")
            query = fit_image(query_path, image_size)
            sheet.paste(query, (margin, top))
            draw.rectangle((margin, top, margin + image_size[0], top + image_size[1]), outline="#222222", width=2)
            text_lines(draw, (margin, top + 154), [f"query #{sample_index} | true: {item['true_class']}", Path(item["relative_image"]).name], body_font, spacing=1)
            for col, (key, (name, _)) in enumerate(MODELS.items(), start=1):
                ref_index = item[f"{name}_reference_index"]
                ref_path = args.data_root / records[ref_index]["relative_image"]
                if file_hash(ref_path) != records[ref_index]["sha256"]:
                    raise ValueError(f"Reference image differs from manifest: {ref_index}")
                thumb = fit_image(ref_path, image_size)
                left = margin + col * cell_width
                sheet.paste(thumb, (left, top))
                correct = item[f"{name}_correct"] == "True"
                color = "#138a36" if correct else "#c62828"
                draw.rectangle((left, top, left + image_size[0], top + image_size[1]), outline=color, width=4)
                lines = [
                    f"pred: {item[f'{name}_predicted']} | ref #{ref_index}",
                    f"cos={float(item[f'{name}_cosine_similarity']):.4f} | {'correct' if correct else 'wrong'}",
                ]
                text_lines(draw, (left, top + 154), lines, small_font, fill=color, spacing=2)
        sheet_path = args.output_dir / f"pattern_{pattern}.png"
        sheet.save(sheet_path, format="PNG", optimize=True)

    summary = {
        "manifest_sha256": file_hash(manifest_path),
        "prediction_csv_sha256": source_hashes,
        "models": {key: name for key, (name, _) in MODELS.items()},
        "correct_pattern_counts_DRC": dict(sorted(patterns.items())),
        "validation_count": len(query_ids),
        "union_error_count": len(audit),
        "output_csv": csv_path.name,
        "contact_sheets": sorted(path.name for path in args.output_dir.glob("pattern_*.png")),
        "note": "Local derived thumbnails only; do not upload because the project excludes source/derived image data.",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Validation images:", len(query_ids))
    print("Images wrong for at least one of D/R/C:", len(audit))
    print("Correctness patterns D/R/C:", dict(sorted(patterns.items())))
    print("Saved audit:", csv_path)
    print("Saved sheets:", len(summary["contact_sheets"]))
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
