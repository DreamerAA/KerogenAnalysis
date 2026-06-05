from pathlib import Path
from typing import Iterable

from utils.utils import kprint


class TrajectoryStepsInfo:
    def __init__(self):
        import re

        self.pattern = re.compile(
            r"t=\s*([0-9]+(?:\.[0-9]+)?)\s+step=\s*([0-9]+)"
        )
        self.steps = []
        self.times = []
        self.delta = -1

    def get_step(self, line: str) -> None:
        match = self.pattern.search(line)
        if match is None:
            return

        t = int(float(match.group(1)))
        step = int(match.group(2))
        self.steps.append(step)
        self.times.append(t)


def parse_atom_line(line: str) -> tuple[int, str] | None:
    """
    Parse a .gro-like atom line and return residue id and residue name.
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
    """
    if len(line) < 20:
        return line

    return line[:15] + f"{atom_number % 100000:5d}" + line[20:]


def filter_trajectory(
    input_path: Path,
    output_path: Path,
    include_selections: Iterable[tuple[str, int]] | None = None,
    exclude_resnames: Iterable[str] | None = None,
    keep_original_count: bool = False,
    renumber_atoms: bool = False,
    drop_empty_frames: bool = False,
    skip_existing: bool = True,
) -> TrajectoryStepsInfo:
    frame_count = 0
    written_frame_count = 0
    total_kept_atoms = 0
    info = TrajectoryStepsInfo()

    include_set = set(include_selections or [])
    exclude_set = set(exclude_resnames or [])

    if include_set and exclude_set:
        raise ValueError("Use include_selections or exclude_resnames, not both")

    if skip_existing and output_path.exists():
        print(f"Output file {output_path} already exists. Skipping.")
        return info

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        while True:
            header = fin.readline()

            if not header:
                break

            if info.pattern.search(header):
                info.get_step(header)
            count_line = fin.readline()
            if not count_line:
                raise RuntimeError(
                    f"Unexpected EOF after frame header: {header.strip()}"
                )

            try:
                n_atoms = int(count_line.strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot parse atom count after header: {header.strip()}\n"
                    f"Atom count line was: {count_line!r}"
                ) from exc

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

                if include_set and (resname, resid) not in include_set:
                    continue
                if exclude_set and resname in exclude_set:
                    continue

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
    return info
