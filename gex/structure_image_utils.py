import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

additional_radius = 0.0

atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}

ext_radius = {
    i: s
    for i, s in enumerate(
        [
            additional_radius,
            additional_radius,
            additional_radius,
            0.0,
            additional_radius,
        ]
    )
}

STRUCTURE_PATTERN = re.compile(
    r"struct-num=(?P<step>\d+)"
    r"_time-ps=(?P<time_ps>\d+(?:\.\d+)?)\.pickle"
)

HEADER_PATTERN = re.compile(
    r"t=\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s+step=\s*(?P<step>[0-9]+)"
)


@dataclass(frozen=True)
class GroTrajectoryInfo:
    start_step: int
    step_size: int | None
    start_time: int
    time_step_size: int | None
    frame_count: int | None
    last_step: int | None


def get_size(type_id: int) -> float:
    return atom_real_sizes[type_id]


def get_ext_size(type_id: int) -> float:
    return ext_radius[type_id]


def kprint(message: str) -> None:
    print(f" --- {message}")


def write_binary_file(array, file_name: Path) -> None:
    with file_name.open("wb") as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))


def parse_gro_header(line: str) -> tuple[int, int]:
    match = HEADER_PATTERN.search(line)
    if match is None:
        raise ValueError(f"Cannot parse trajectory header: {line.strip()}")
    time_ps = int(float(match.group("time")))
    step = int(match.group("step"))
    return step, time_ps


def scan_gro_trajectory_info(
    input_path: Path,
    count_all_frames: bool = True,
) -> GroTrajectoryInfo:
    steps: list[int] = []
    times: list[int] = []
    frame_count = 0

    with input_path.open("r", encoding="utf-8") as f:
        while True:
            header = f.readline()
            if not header:
                break

            step, time_ps = parse_gro_header(header)
            count_line = f.readline()
            if not count_line:
                raise RuntimeError(
                    f"Unexpected EOF after frame header: {header.strip()}"
                )
            try:
                atom_count = int(count_line.strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot parse atom count after header: {header.strip()}"
                ) from exc

            for _ in range(atom_count + 1):
                if not f.readline():
                    raise RuntimeError(
                        f"Unexpected EOF while skipping frame {frame_count + 1}"
                    )

            frame_count += 1
            if len(steps) < 2:
                steps.append(step)
                times.append(time_ps)

            if not count_all_frames and len(steps) == 2:
                break

    if not steps:
        raise RuntimeError(f"No frames found in {input_path}")

    step_size = steps[1] - steps[0] if len(steps) > 1 else None
    time_step_size = times[1] - times[0] if len(times) > 1 else None
    counted_frames = frame_count if count_all_frames else None
    last_step = (
        steps[0] + (counted_frames - 1) * step_size
        if counted_frames is not None and step_size is not None
        else None
    )
    return GroTrajectoryInfo(
        start_step=steps[0],
        step_size=step_size,
        start_time=times[0],
        time_step_size=time_step_size,
        frame_count=counted_frames,
        last_step=last_step,
    )


def generate_indexes_by_mode(
    start_step: int,
    step_size: int,
    full_count_steps: int,
    count_structures: int,
    mode: str,
) -> list[int]:
    if count_structures <= 0:
        raise ValueError("--count-structures must be positive")

    last_step = full_count_steps * step_size + start_step
    if mode == "all":
        step = max(1, int(full_count_steps / count_structures))
    elif mode == "part":
        step = 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    indexes = [
        start_step + step_size * i * step
        for i in range(count_structures)
        if start_step + step_size * i * step <= last_step
    ]

    if mode == "all" and last_step not in indexes:
        indexes.append(last_step)

    return indexes


def parse_structure_file_info(path: Path) -> tuple[int, float] | None:
    match = STRUCTURE_PATTERN.fullmatch(path.name)
    if match is None:
        return None

    return int(match.group("step")), float(match.group("time_ps"))


def list_available_structure_indexes(structures_dir: Path) -> list[int]:
    indexes = []
    for path in structures_dir.iterdir():
        if not path.is_file():
            continue
        parsed = parse_structure_file_info(path)
        if parsed is None:
            continue
        step, _ = parsed
        indexes.append(step)

    return sorted(dict.fromkeys(indexes))


def generate_indexes_from_available_structures(
    structures_dir: Path,
    mode: str,
    count_slices: int,
) -> list[int]:
    if count_slices <= 0:
        raise ValueError("--count-slices must be positive")

    available_indexes = list_available_structure_indexes(structures_dir)
    if not available_indexes:
        raise RuntimeError(f"No structure pickle files found in {structures_dir}")

    if mode == "part":
        return available_indexes[:count_slices]
    if mode != "all":
        raise ValueError(f"Unknown mode: {mode}")

    if count_slices >= len(available_indexes):
        return available_indexes

    step = max(1, int(len(available_indexes) / count_slices))
    indexes = available_indexes[::step][:count_slices]
    if available_indexes[-1] not in indexes:
        indexes.append(available_indexes[-1])
    return indexes


def collect_processing_indexes(
    structures_dir: Path,
    indexes: list[str],
    indexes_file: Path | None,
    mode: str | None,
    count_slices: int | None,
) -> list[int] | None:
    has_explicit_indexes = bool(indexes or indexes_file)
    has_mode_indexes = bool(mode or count_slices is not None)

    if has_explicit_indexes and has_mode_indexes:
        raise ValueError("Use either --index/--indexes-file or --mode/--count-slices")
    if has_explicit_indexes:
        return collect_indexes(indexes, indexes_file)
    if has_mode_indexes:
        if mode is None or count_slices is None:
            raise ValueError("--mode and --count-slices must be used together")
        return generate_indexes_from_available_structures(
            structures_dir=structures_dir,
            mode=mode,
            count_slices=count_slices,
        )

    return None


def parse_indexes(values: Iterable[str] | None) -> list[int]:
    indexes: list[int] = []
    for value in values or []:
        for item in value.replace(",", " ").split():
            indexes.append(int(item))
    return indexes


def read_indexes_file(path: Path) -> list[int]:
    return parse_indexes(path.read_text(encoding="utf-8").splitlines())


def collect_indexes(indexes: list[str], indexes_file: Path | None) -> list[int]:
    result = list(indexes)
    if indexes_file is not None:
        result.extend(read_indexes_file(indexes_file))

    unique_result = list(dict.fromkeys(result))
    if not unique_result:
        raise ValueError("Pass at least one --index or --indexes-file")
    return unique_result


def structure_file_name(num: int, time_ps: int | float) -> str:
    return f"struct-num={num}_time-ps={time_ps}.pickle"


def image_base_name(
    num: int,
    time_ps: int | float,
    bbox,
    resolution: float,
) -> str:
    return (
        f"result-img-num={num}_time-ps={time_ps}"
        f"_bbox={bbox._short_str()}_resolution={resolution:.9f}"
    )


def iter_structure_files(
    structures_dir: Path,
    indexes: Iterable[int] | None = None,
) -> list[Path]:
    index_set = set(indexes or [])
    filenames = []

    for path in structures_dir.iterdir():
        if not path.is_file():
            continue
        match = STRUCTURE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        if index_set and int(match.group("step")) not in index_set:
            continue
        filenames.append(path)

    return sorted(filenames)


def load_structure(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def build_segmentator(structure, ref_size: int, dev: float):
    from base.kerogendata import KerogenData
    from base.periodizer import Periodizer
    from processes.segmentaion import Segmentator

    num, time_ps, atoms, size = structure
    bbox = Segmentator.cut_cell(size, dev)
    resolution = min(bbox.size()) / ref_size
    img_size = Segmentator.calc_image_size(
        bbox.size(), reference_size=ref_size, by_min=True
    )

    kerogen_data = KerogenData(None, atoms, bbox)
    if not kerogen_data.checkPeriodization():
        Periodizer.periodize(kerogen_data)

    segmentator = Segmentator(
        kerogen_data,
        img_size,
        size_data=get_size,
        radius_extention=get_ext_size,
        partitioning=2,
    )
    return num, time_ps, bbox, resolution, segmentator
