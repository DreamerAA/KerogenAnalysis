from pathlib import Path
import argparse

from utils.gro_trajectory import TrajectoryStepsInfo, filter_trajectory


def run(
    data_path: str,
    save_path: str,
    exclude_resnames: set[str] | None = None,
    keep_original_count: bool = False,
    renumber_atoms: bool = False,
    drop_empty_frames: bool = False,
) -> TrajectoryStepsInfo:
    return filter_trajectory(
        input_path=Path(data_path),
        output_path=Path(save_path),
        exclude_resnames=exclude_resnames or {"KRG"},
        keep_original_count=keep_original_count,
        renumber_atoms=renumber_atoms,
        drop_empty_frames=drop_empty_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove selected residue names from a .gro-like trajectory."
    )

    parser.add_argument("input", type=Path, help="Input trajectory file")
    parser.add_argument("output", type=Path, help="Output trajectory file")
    parser.add_argument(
        "--exclude-resname",
        action="append",
        help="Residue name to remove. Can be used multiple times.",
    )
    parser.add_argument(
        "--keep-original-count",
        action="store_true",
        help="Keep original atom count line. Usually not recommended.",
    )
    parser.add_argument(
        "--renumber-atoms",
        action="store_true",
        help="Renumber atom serial numbers inside each filtered frame.",
    )
    parser.add_argument(
        "--drop-empty-frames",
        action="store_true",
        help="Do not write frames where no atoms remain.",
    )

    args = parser.parse_args()
    run(
        data_path=str(args.input),
        save_path=str(args.output),
        exclude_resnames=set(args.exclude_resname or ["KRG"]),
        keep_original_count=args.keep_original_count,
        renumber_atoms=args.renumber_atoms,
        drop_empty_frames=args.drop_empty_frames,
    )


if __name__ == '__main__':
    main()
