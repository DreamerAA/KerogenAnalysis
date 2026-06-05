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
)


def build_distance_maps(
    structures_dir: Path,
    output_dir: Path,
    indexes: list[int] | None,
    ref_size: int,
    dev: float,
) -> None:
    float_image_dir = output_dir / "float_images"
    float_image_dir.mkdir(parents=True, exist_ok=True)

    structure_files = iter_structure_files(structures_dir, indexes)
    for i, structure_file in enumerate(structure_files):
        structure = load_structure(structure_file)
        num, time_ps, bbox, resolution, segmentator = build_segmentator(
            structure, ref_size=ref_size, dev=dev
        )
        base_name = image_base_name(num, time_ps, bbox, resolution)
        float_path = float_image_dir / f"{base_name}.npy"

        if float_path.exists():
            kprint(
                f"Skip distance map with num={num}. Its {i + 1} from {len(structure_files)}"
            )
            continue

        kprint(f"Run distance map for num={num}")
        start_time = time.time()
        float_img = segmentator.dist_map()
        np.save(float_path, float_img)
        kprint(
            f"Distance map struct {num} is finished! Elapsed time: {time.time() - start_time}s. Its {i + 1} from {len(structure_files)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build distance maps for extracted structure pickle files."
    )
    parser.add_argument("structures_dir", type=Path, help="Input structures dir")
    parser.add_argument("output_dir", type=Path, help="Output base directory")
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
    parser.add_argument("--dev", type=float, default=4.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected indexes and exit without building distance maps.",
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

    build_distance_maps(
        structures_dir=args.structures_dir,
        output_dir=args.output_dir,
        indexes=indexes,
        ref_size=args.ref_size,
        dev=args.dev,
    )


if __name__ == "__main__":
    main()
