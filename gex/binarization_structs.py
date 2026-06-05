import argparse
import time
from pathlib import Path

import numpy as np

from gex.structure_image_utils import (
    build_segmentator,
    collect_processing_indexes,
    image_base_name,
    iter_structure_files,
    kprint,
    load_structure,
    write_binary_file,
)


def binarize_structures(
    structures_dir: Path,
    output_bin_dir: Path,
    output_raw_dir: Path,
    indexes: list[int] | None,
    ref_size: int,
    dev: float,
    num_workers: int,
) -> None:
    output_bin_dir.mkdir(parents=True, exist_ok=True)
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    structure_files = iter_structure_files(structures_dir, indexes)
    for i, structure_file in enumerate(structure_files):
        structure = load_structure(structure_file)
        num, time_ps, bbox, resolution, segmentator = build_segmentator(
            structure, ref_size=ref_size, dev=dev
        )
        base_name = image_base_name(num, time_ps, bbox, resolution)
        binarized_path = output_bin_dir / f"{base_name}.npy"
        raw_path = output_raw_dir / f"{base_name}.raw"

        if binarized_path.exists() and raw_path.exists():
            kprint(
                f"Skip binarization with num={num}. Its {i + 1} from {len(structure_files)}"
            )
            continue

        kprint(f"Run binarization for num={num}")
        start_time = time.time()
        img = 1 - segmentator.binarize(num_workers=num_workers)
        np.save(binarized_path, img)
        write_binary_file(img, raw_path)
        kprint(
            f"Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s. Its {i + 1} from {len(structure_files)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize extracted structure pickle files."
    )
    parser.add_argument(
        "structures_dir", type=Path, help="Input structures dir"
    )
    parser.add_argument(
        "output_bin_dir", type=Path, help="Output bin image directory"
    )
    parser.add_argument(
        "output_raw_dir", type=Path, help="Output raw image directory"
    )
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help="Structure step number. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--indexes-file",
        type=Path,
        help="Text file with structure step numbers separated by whitespace or commas.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "part"],
        help="Select structures from structures_dir by old-style mode.",
    )
    parser.add_argument(
        "--count-slices",
        type=int,
        help="How many structures to select with --mode.",
    )
    parser.add_argument("--ref-size", type=int, required=True)
    parser.add_argument("--dev", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected indexes and exit without binarization.",
    )

    args = parser.parse_args()
    try:
        indexes = collect_processing_indexes(
            structures_dir=args.structures_dir,
            indexes=args.index,
            indexes_file=args.indexes_file,
            mode=args.mode,
            count_slices=args.count_slices,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        if indexes is None:
            kprint("Selected all available structures")
        else:
            preview = ", ".join(str(index) for index in indexes[:20])
            if len(indexes) > 20:
                preview += ", ..."
            kprint(f"Selected structure indexes: {len(indexes)}")
            kprint(f"Indexes preview: {preview}")
        return

    binarize_structures(
        structures_dir=args.structures_dir,
        output_bin_dir=args.output_bin_dir,
        output_raw_dir=args.output_raw_dir,
        indexes=indexes,
        ref_size=args.ref_size,
        dev=args.dev,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
