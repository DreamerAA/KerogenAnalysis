import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from gex.structure_image_utils import (
    collect_indexes,
    generate_indexes_by_mode,
    kprint,
    scan_gro_trajectory_info,
    structure_file_name,
)


def extract_structures(
    input_path: Path,
    output_dir: Path,
    indexes: list[int],
    slice_len: int = 100,
) -> None:
    from base.reader import Reader

    output_dir.mkdir(parents=True, exist_ok=True)

    aindexes = np.asarray(indexes, dtype=np.int32)
    existing_mask = np.asarray(
        [
            any(output_dir.glob(f"struct-num={num}_time-ps=*.pickle"))
            for num in aindexes
        ],
        dtype=bool,
    )
    aindexes = aindexes[~existing_mask]

    count_steps = len(aindexes) // slice_len + 1
    for i in range(count_steps):
        start_time = time.time()
        start = i * slice_len
        stop = min((i + 1) * slice_len, len(aindexes))
        cur_indexes = aindexes[start:stop].tolist()
        if not cur_indexes:
            continue

        structures = Reader.read_structures_by_num(str(input_path), cur_indexes)
        for struct in structures:
            num, time_ps, _, _ = struct
            save_path = output_dir / structure_file_name(num, time_ps)
            with save_path.open("wb") as f:
                pickle.dump(struct, f)

        kprint(f"Count structures step: {i + 1} from {count_steps}")
        kprint(f"Reading finished! Elapsed time: {time.time() - start_time}s")


def build_indexes_from_args(args: argparse.Namespace) -> list[int]:
    if args.auto_indexes:
        info = scan_gro_trajectory_info(
            args.input,
            count_all_frames=args.full_count_steps is None,
        )
        if info.step_size is None:
            raise ValueError("Need at least two frames to infer step size")

        full_count_steps = args.full_count_steps
        if full_count_steps is None:
            if info.frame_count is None:
                raise ValueError("Cannot infer full_count_steps")
            full_count_steps = info.frame_count - 1

        indexes = generate_indexes_by_mode(
            start_step=info.start_step,
            step_size=info.step_size,
            full_count_steps=full_count_steps,
            count_structures=args.count_structures,
            mode=args.mode,
        )
        kprint(
            "Trajectory info: "
            f"start_step={info.start_step}, "
            f"step_size={info.step_size}, "
            f"start_time={info.start_time}, "
            f"time_step_size={info.time_step_size}, "
            f"full_count_steps={full_count_steps}"
        )
        kprint(f"Generated structure indexes: {len(indexes)}")
        return indexes

    return collect_indexes(args.index, args.indexes_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected structures from a .gro trajectory into pickle files."
    )
    parser.add_argument("input", type=Path, help="Input .gro trajectory file")
    parser.add_argument("output_dir", type=Path, help="Directory for structures")
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
        "--auto-indexes",
        action="store_true",
        help="Infer trajectory parameters and generate indexes by mode/count.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "part"],
        default="all",
        help="Old-style index generation mode for --auto-indexes.",
    )
    parser.add_argument(
        "--count-structures",
        type=int,
        help="How many structures to request in --auto-indexes mode.",
    )
    parser.add_argument(
        "--full-count-steps",
        type=int,
        help=(
            "Old full_count_steps value. If omitted, frames are counted by "
            "quickly skipping through the trajectory."
        ),
    )
    parser.add_argument(
        "--slice-len",
        type=int,
        default=100,
        help="How many requested structures to read per batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated indexes and exit without extracting structures.",
    )

    args = parser.parse_args()
    if args.auto_indexes and args.count_structures is None:
        parser.error("--auto-indexes requires --count-structures")
    if args.auto_indexes and (args.index or args.indexes_file):
        parser.error("Use either --auto-indexes or --index/--indexes-file")

    indexes = build_indexes_from_args(args)
    if args.dry_run:
        preview = ", ".join(str(index) for index in indexes[:20])
        if len(indexes) > 20:
            preview += ", ..."
        kprint(f"Indexes preview: {preview}")
        return

    extract_structures(args.input, args.output_dir, indexes, args.slice_len)


if __name__ == "__main__":
    main()
