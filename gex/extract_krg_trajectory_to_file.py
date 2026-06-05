#!/usr/bin/env python3

from pathlib import Path
import argparse

from utils.gro_trajectory import filter_trajectory as filter_gro_trajectory


def parse_selection(value: str):
    """
    Parse selection in format GAS:NUM.

    Example:
        KRG:99
        CH4:119
        H2:10
    """
    try:
        gas, num = value.split(":")
        gas = gas.strip()
        num = int(num.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid selection '{value}'. Expected format GAS:NUM, for example KRG:99"
        )

    if not gas:
        raise argparse.ArgumentTypeError("Gas/residue name must not be empty")

    return gas, num


def main():
    parser = argparse.ArgumentParser(
        description="Filter .gro-like trajectory by molecule/residue number and residue name."
    )

    parser.add_argument("input", type=Path, help="Input trajectory file")
    parser.add_argument(
        "output", type=Path, help="Output filtered trajectory file"
    )

    parser.add_argument(
        "--select",
        action="append",
        type=parse_selection,
        required=True,
        help="Selection in format GAS:NUM, for example KRG:99. Can be used multiple times.",
    )

    parser.add_argument(
        "--keep-original-count",
        action="store_true",
        help="Keep original atom count line. Usually not recommended for valid .gro-like output.",
    )

    parser.add_argument(
        "--renumber-atoms",
        action="store_true",
        help="Renumber atom serial numbers inside each filtered frame.",
    )

    parser.add_argument(
        "--drop-empty-frames",
        action="store_true",
        help="Do not write frames where no atoms matched the selection.",
    )

    args = parser.parse_args()

    selections = set(args.select)

    filter_gro_trajectory(
        input_path=args.input,
        output_path=args.output,
        include_selections=selections,
        keep_original_count=args.keep_original_count,
        renumber_atoms=args.renumber_atoms,
        drop_empty_frames=args.drop_empty_frames,
    )


if __name__ == "__main__":
    main()
