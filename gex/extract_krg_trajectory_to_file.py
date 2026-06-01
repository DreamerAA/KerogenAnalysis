#!/usr/bin/env python3

from pathlib import Path
import argparse

from utils.utils import kprint


def parse_atom_line(line: str):
    """
    Parse .gro-like atom line.

    Example:
        '   99KRG     c3    1   1.632   1.179   5.537'
         resid = 99
         resname = KRG
    """
    if len(line) < 10:
        return None

    resid_str = line[0:5].strip()
    resname = line[5:10].strip()

    if not resid_str.isdigit():
        return None

    return int(resid_str), resname


def set_atom_number(line: str, atom_number: int) -> str:
    """
    Replace atom serial number in columns 16-20 of a .gro-like line.

    GRO atom number field is usually columns [15:20].
    """
    if len(line) < 20:
        return line

    return line[:15] + f"{atom_number % 100000:5d}" + line[20:]


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


def filter_trajectory(
    input_path: Path,
    output_path: Path,
    selections: set[tuple[str, int]],
    keep_original_count: bool = False,
    renumber_atoms: bool = False,
    drop_empty_frames: bool = False,
):
    frame_count = 0
    written_frame_count = 0
    total_kept_atoms = 0

    if output_path.exists():
        kprint(f"Output file {output_path} already exists. Skipping.")
        return

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        while True:
            header = fin.readline()

            if not header:
                break

            count_line = fin.readline()
            if not count_line:
                raise RuntimeError(
                    f"Unexpected EOF after frame header: {header.strip()}"
                )

            try:
                n_atoms = int(count_line.strip())
            except ValueError:
                raise RuntimeError(
                    f"Cannot parse atom count after header: {header.strip()}\n"
                    f"Atom count line was: {count_line!r}"
                )

            frame_count += 1
            kept_lines = []

            for _ in range(n_atoms):
                atom_line = fin.readline()
                if not atom_line:
                    raise RuntimeError(
                        f"Unexpected EOF while reading atoms in frame {frame_count}"
                    )

                parsed = parse_atom_line(atom_line)
                if parsed is None:
                    continue

                resid, resname = parsed

                if (resname, resid) in selections:
                    kept_lines.append(atom_line)

            box_line = fin.readline()
            if not box_line:
                raise RuntimeError(
                    f"Unexpected EOF while reading box line in frame {frame_count}"
                )

            if drop_empty_frames and not kept_lines:
                continue

            if renumber_atoms:
                kept_lines = [
                    set_atom_number(line, i + 1)
                    for i, line in enumerate(kept_lines)
                ]

            fout.write(header)

            if keep_original_count:
                fout.write(count_line)
            else:
                fout.write(f"{len(kept_lines):5d}\n")

            fout.writelines(kept_lines)
            fout.write(box_line)

            written_frame_count += 1
            total_kept_atoms += len(kept_lines)

    kprint(f"Read frames: {frame_count}")
    kprint(f"Written frames: {written_frame_count}")
    kprint(f"Total kept atom lines: {total_kept_atoms}")
    kprint(f"Output: {output_path}")


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

    filter_trajectory(
        input_path=args.input,
        output_path=args.output,
        selections=selections,
        keep_original_count=args.keep_original_count,
        renumber_atoms=args.renumber_atoms,
        drop_empty_frames=args.drop_empty_frames,
    )


if __name__ == "__main__":
    main()
