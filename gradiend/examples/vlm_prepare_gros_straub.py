"""Download and pair Gros-Straub portrait attributes into a ``pairs.json`` manifest.

Fetches matched ``vanilla`` vs. ``hat`` or ``glasses`` photographs from the four
public Mendeley Data archives (``bpy24trj6s``, ``ycjd7mdsbs``, ``55wmmr8j3g``,
``zsv3bgp7t4``) using HTTP range requests so the multi-gigabyte source ZIPs do
not need to be stored on disk.

Example usage:

    python -m gradiend.examples.vlm_prepare_gros_straub --concept glasses --output data/gros_straub_glasses
    python -m gradiend.examples.vlm_prepare_gros_straub --concept hat --output data/gros_straub_hats
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from time import sleep
import zipfile

from PIL import Image, ImageOps


ARCHIVE_IDS = ("bpy24trj6s", "ycjd7mdsbs", "55wmmr8j3g", "zsv3bgp7t4")
API = "https://data.mendeley.com/public-api/datasets"
MAX_BLOCK = 52 * 1024 * 1024
JPEG_QUALITY = 92
POSES = tuple(f"pos_{i}" for i in range(5))
LIGHT_POSITIONS = tuple(f"angle_{i}" for i in range(1, 8))
IMAGE_NAME = re.compile(r"IMG_(\d+)\.JPE?G", re.IGNORECASE)

CONCEPT_TEXT = {
    "hat": {
        "prompt": "Is the person wearing a hat? Answer yes or no.",
        "factual_answer": "yes",
        "counterfactual_answer": "no",
        "factual_prompt": "A realistic head-and-shoulders photograph of an adult wearing a hat.",
        "counterfactual_prompt": "A realistic head-and-shoulders photograph of an adult without a hat.",
    },
    "glasses": {
        "prompt": "Is the person wearing eyeglasses? Answer yes or no.",
        "factual_answer": "yes",
        "counterfactual_answer": "no",
        "factual_prompt": "A realistic head-and-shoulders photograph of an adult wearing eyeglasses.",
        "counterfactual_prompt": "A realistic head-and-shoulders photograph of an adult without eyeglasses.",
    },
}


def _capture_order(info: zipfile.ZipInfo) -> int:
    match = IMAGE_NAME.fullmatch(PurePosixPath(info.filename).name)
    if match is None:
        raise ValueError(f"Unexpected capture filename: {info.filename}")
    return int(match.group(1))


def _curl(url: str, *options: str) -> bytes:
    last_error = ""
    for attempt in range(4):
        command = [
            "curl", "--fail", "--silent", "--show-error", "--location",
            "--connect-timeout", "15", "--max-time", "90", *options, url,
        ]
        completed = subprocess.run(command, capture_output=True)
        if completed.returncode == 0:
            return completed.stdout
        last_error = completed.stderr.decode(errors="replace").strip()
        if attempt < 3:
            sleep(2 ** attempt)
    raise RuntimeError(f"Failed fetching {url}: {last_error}")


class _RangeReader(io.RawIOBase):
    """Seekable ZIP reader backed by bounded HTTP range requests."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.start = 0
        self.block = b""

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        position = (0, self.position, self.size)[whence] + offset
        if position < 0:
            raise OSError("Cannot seek before archive start")
        self.position = position
        return position

    def fetch(self, start: int, end: int) -> None:
        length = end - start
        if not 0 < length <= MAX_BLOCK or not 0 <= start < end <= self.size:
            raise ValueError(f"Invalid archive range {start}-{end - 1} for {self.size} bytes")
        self.block = b""
        payload = _curl(
            self.url,
            "--range", f"{start}-{end - 1}",
            "--max-filesize", str(max(length, 1024)),
            "--write-out", "\n%{http_code}\n%header{content-range}\n%{url_effective}",
        )
        block, status, content_range, url = payload.rsplit(b"\n", 3)
        expected = f"bytes {start}-{end - 1}/{self.size}".encode()
        if status != b"206" or content_range != expected or len(block) != length:
            raise ValueError(f"Archive response {status!r}/{content_range!r} did not match {expected!r}")
        self.start, self.block, self.url = start, block, url.decode()

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.position + size, self.size)
        if end <= self.position:
            return b""
        if not self.start <= self.position < end <= self.start + len(self.block):
            self.fetch(self.position, end)
        data = self.block[self.position - self.start:end - self.start]
        self.position = end
        return data


def _archive_reader(dataset: str) -> _RangeReader:
    files = json.loads(_curl(f"{API}/{dataset}/files?folder_id=root&version=1"))
    zips = [item for item in files if item["filename"].endswith(".zip")]
    if len(zips) != 1:
        raise ValueError(f"Expected 1 ZIP in Mendeley dataset {dataset}, found {len(zips)}")
    content = zips[0]["content_details"]
    return _RangeReader(content["download_url"], content["size"])


def _regions(archive: zipfile.ZipFile, entries: list[tuple[zipfile.ZipInfo, Path]], archive_size: int):
    directory = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
    ends = {entry.header_offset: following.header_offset for entry, following in zip(directory, directory[1:])}
    ends[directory[-1].header_offset] = archive_size
    block = []
    for info, path in sorted(entries, key=lambda item: item[0].header_offset):
        start, end = info.header_offset, ends[info.header_offset]
        if block and (start - block[-1][1] > 4096 or end - block[0][0] > MAX_BLOCK):
            yield block
            block = []
        block.append((start, end, info, path))
    if block:
        yield block


def _write_photo(source: bytes, path: Path) -> None:
    with Image.open(io.BytesIO(source)) as image:
        oriented = ImageOps.exif_transpose(image).convert("RGB")
    oriented.thumbnail((768, 768), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    oriented.save(temporary, format="JPEG", quality=JPEG_QUALITY)
    temporary.replace(path)


def _assign_splits(groups: list[str]) -> dict[str, str]:
    ordered = sorted(groups)
    if len(ordered) < 3:
        raise ValueError(f"Need at least 3 subjects for train/validation/test splits, found {len(ordered)}")
    n_test = max(1, round(len(ordered) * 3 / 11))
    n_val = max(1, round(len(ordered) * 2 / 11))
    if n_test + n_val >= len(ordered):
        n_test = 1
        n_val = 1
    n_train = len(ordered) - n_val - n_test
    splits = {}
    for index, group in enumerate(ordered):
        if index < n_train:
            splits[group] = "train"
        elif index < n_train + n_val:
            splits[group] = "validation"
        else:
            splits[group] = "test"
    return splits


def prepare_gros_straub(
    output: Path,
    *,
    concept: str = "glasses",
    max_pairs_per_subject: int | None = None,
) -> Path:
    """Download matched Gros-Straub pairs and write ``pairs.json`` under ``output``."""
    if concept not in CONCEPT_TEXT:
        raise ValueError(f"Unsupported concept {concept!r}; expected one of {sorted(CONCEPT_TEXT)}")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    readers = {}
    archives = {}
    blocks_by_dataset = {}
    all_groups = set()
    for dataset in ARCHIVE_IDS:
        reader = _archive_reader(dataset)
        archive = zipfile.ZipFile(reader)
        blocks = defaultdict(lambda: defaultdict(list))
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 6 or not parts[-5].isdigit() or not info.filename.lower().endswith((".jpg", ".jpeg")):
                continue
            subject, condition, pose, lighting, filename = parts[-5:]
            if pose not in POSES or lighting not in LIGHT_POSITIONS:
                continue
            if not IMAGE_NAME.fullmatch(filename):
                continue
            group = f"gros-{dataset}-{subject}"
            blocks[group, pose, lighting][condition].append(info)
            all_groups.add(group)
        readers[dataset] = reader
        archives[dataset] = archive
        blocks_by_dataset[dataset] = blocks

    splits = _assign_splits(list(all_groups))
    manifest_pairs = []
    for dataset in ARCHIVE_IDS:
        reader = readers[dataset]
        archive = archives[dataset]
        blocks = blocks_by_dataset[dataset]
        downloads = []
        subject_counts = defaultdict(int)
        dataset_groups = sorted({group for group, _, _ in blocks})
        for group in dataset_groups:
            for pose in POSES:
                for lighting in LIGHT_POSITIONS:
                    if max_pairs_per_subject is not None and subject_counts[group] >= max_pairs_per_subject:
                        break
                    conditions = blocks[group, pose, lighting]
                    vanilla = conditions.get("vanilla", [])
                    target = conditions.get(concept, [])
                    if len(vanilla) != 5 or len(target) != 5:
                        continue
                    a_order = sorted(vanilla, key=_capture_order)
                    b_order = sorted(target, key=_capture_order)
                    for index, (neg_info, pos_info) in enumerate(zip(a_order, b_order)):
                        if max_pairs_per_subject is not None and subject_counts[group] >= max_pairs_per_subject:
                            break
                        prefix = f"{group.removeprefix('gros-')}-{pose}-{lighting}-{index}"
                        neg_path = output / "images" / f"{prefix}-vanilla.jpg"
                        pos_path = output / "images" / f"{prefix}-{concept}.jpg"
                        if not neg_path.is_file():
                            downloads.append((neg_info, neg_path))
                        if not pos_path.is_file():
                            downloads.append((pos_info, pos_path))
                        manifest_pairs.append({
                            "group_id": group,
                            "split": splits[group],
                            "factual_image": pos_path.relative_to(output).as_posix(),
                            "counterfactual_image": neg_path.relative_to(output).as_posix(),
                        })
                        subject_counts[group] += 1

        if downloads:
            print(f"Downloading {len(downloads)} images from {dataset}...", flush=True)
            for batch in _regions(archive, downloads, reader.size):
                reader.fetch(batch[0][0], batch[-1][1])
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda item: _write_photo(archive.read(item[2]), item[3]), batch))
        archive.close()

    manifest = {**CONCEPT_TEXT[concept], "pairs": manifest_pairs}
    manifest_path = output / "pairs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest_pairs)} {concept} pairs across {len(splits)} subjects to {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", choices=tuple(CONCEPT_TEXT), default="glasses")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-pairs-per-subject", type=int, default=None)
    args = parser.parse_args()
    output = args.output or Path("data") / f"gros_straub_{args.concept}"
    prepare_gros_straub(output, concept=args.concept, max_pairs_per_subject=args.max_pairs_per_subject)


if __name__ == "__main__":
    main()
