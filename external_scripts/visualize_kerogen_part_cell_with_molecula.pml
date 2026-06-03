reinitialize

python
import argparse
import os
from pathlib import Path
import shlex
import sys

from pymol import cmd


def parse_args():
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not script_args:
        script_args = shlex.split(os.environ.get("PYMOL_SCRIPT_ARGS", ""))
    parser = argparse.ArgumentParser(
        description=(
            "Render a small simulation-cell fragment and highlight one molecule. "
            "The PDB file is used as the bonded topology; GRO coordinates are "
            "taken from the selected frame."
        )
    )
    parser.add_argument("--ker-pdb", required=True, help="Kerogen PDB with bonds/topology")
    parser.add_argument("--sim-gro", required=True, help="Simulation GRO file")
    parser.add_argument("--frame", "--step", dest="frame", type=int, default=0)
    parser.add_argument(
        "--mol-index",
        "--mol-id",
        dest="mol_index",
        required=True,
        help="Molecule/residue number to highlight, e.g. 15 from the printed subbox list",
    )
    parser.add_argument("--box-size", type=float, default=20.0, help="Subbox side in Angstrom")
    parser.add_argument(
        "--resname",
        default="KRG",
        help="Residue name to extract from GRO when the simulation contains non-kerogen atoms",
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Subbox center in Angstrom. Defaults to the simulation cell center.",
    )
    parser.add_argument("--output", default=None, help="Optional PNG output path")
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument(
        "--camera-rot",
        nargs=3,
        type=float,
        default=(0.0, 110.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Fixed camera rotations in degrees after orienting the subbox.",
    )
    parser.add_argument(
        "--surface-mode",
        choices=("selected", "all", "none"),
        default="all",
        help="Surface rendering mode. 'all' matches visualize_kerogen_part_cell.pml.",
    )
    parser.add_argument(
        "--ray",
        action="store_true",
        help="Use ray tracing for PNG output. Slower, but publication-quality.",
    )
    return parser.parse_args(script_args)


def read_gro_frame(path, frame_index, resname, max_atoms):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        frame = 0
        while True:
            title = f.readline()
            if not title:
                raise ValueError(f"Frame {frame_index} not found in {path}")

            natoms_line = f.readline()
            if not natoms_line:
                raise ValueError(f"Unexpected EOF after GRO title in {path}")

            natoms = int(natoms_line.strip())
            coords = []
            matched_atoms = 0
            for _ in range(natoms):
                line = f.readline()
                if frame != frame_index:
                    continue

                line_resname = line[5:10].strip()
                if resname != "ALL" and line_resname != resname:
                    continue

                matched_atoms += 1
                if len(coords) < max_atoms:
                    # GRO stores coordinates in nm; PyMOL/PDB coordinates are Angstrom.
                    coords.append(
                        (
                            float(line[20:28]) * 10.0,
                            float(line[28:36]) * 10.0,
                            float(line[36:44]) * 10.0,
                        )
                    )

            box_line = f.readline()
            if not box_line:
                raise ValueError(f"Unexpected EOF after GRO atoms in {path}")

            if frame == frame_index:
                box_values = [float(v) * 10.0 for v in box_line.split()]
                if len(box_values) < 3:
                    raise ValueError(f"Cannot parse GRO box line: {box_line!r}")

                return {
                    "title": title.strip(),
                    "coords": coords,
                    "matched_atoms": matched_atoms,
                    "box": tuple(box_values[:3]),
                }

            frame += 1


def selection_from_indices(object_name, indices):
    return f"{object_name} and index " + "+".join(str(index) for index in indices)


args = parse_args()

cmd.load(args.ker_pdb, "kerogen_template")

template_model = cmd.get_model("kerogen_template")
template_atoms = template_model.atom
template_count = len(template_atoms)

frame = read_gro_frame(args.sim_gro, args.frame, args.resname, template_count)
gro_coords = frame["coords"]

if len(gro_coords) < template_count:
    raise ValueError(
        f"Not enough GRO atoms for topology: got {len(gro_coords)}, need {template_count}. "
        f"Check --resname {args.resname!r}. Use --resname ALL only if the GRO atom order "
        f"starts with the kerogen topology atoms."
    )

if frame["matched_atoms"] != template_count:
    print(
        f"Warning: using first {template_count} atoms from {frame['matched_atoms']} GRO atoms "
        f"selected by --resname {args.resname!r}."
    )

coord_by_index = {
    atom.index: coord
    for atom, coord in zip(template_atoms, gro_coords)
}
cmd.alter_state(1, "kerogen_template", "(x, y, z) = coord_by_index[index]", space=locals())

a, b, c = frame["box"]
cmd.set_symmetry("kerogen_template", a, b, c, 90.0, 90.0, 90.0, "P 1")

if args.center is None:
    cx, cy, cz = 0.5 * a, 0.5 * b, 0.5 * c
else:
    cx, cy, cz = args.center

half = 0.5 * args.box_size
xmin, xmax = cx - half, cx + half
ymin, ymax = cy - half, cy + half
zmin, zmax = cz - half, cz + half

sub_selection = (
    f"kerogen_template and x>{xmin} and x<{xmax} "
    f"and y>{ymin} and y<{ymax} and z>{zmin} and z<{zmax}"
)
cmd.select("sub_atoms", sub_selection)
if cmd.count_atoms("sub_atoms") == 0:
    raise ValueError(
        "Selected subbox is empty. Try a larger --box-size or pass --center X Y Z."
    )

cmd.create("subbox_obj", "sub_atoms")
cmd.disable("kerogen_template")
cmd.enable("subbox_obj")

cmd.hide("everything")
cmd.bg_color("white")
cmd.set("orthoscopic", "off")
cmd.set("antialias", 2)
cmd.set("ray_opaque_background", "off")
cmd.set("specular", 0.2)
cmd.set("shininess", 10)
cmd.set("valence", 0)
cmd.set("transparency_mode", 1)

cmd.set_color("base_gray", [0.6, 0.6, 0.6])
cmd.hide("everything", "subbox_obj")

palette = [
    [0.62, 0.53, 0.22],
    [0.45, 0.58, 0.35],
    [0.55, 0.42, 0.30],
    [0.45, 0.50, 0.62],
    [0.62, 0.42, 0.42],
    [0.50, 0.45, 0.62],
]

model = cmd.get_model("subbox_obj")
resis = sorted(set(atom.resi for atom in model.atom), key=lambda x: int(x) if x.isdigit() else x)
if not resis:
    raise ValueError("No molecules/residues found in selected subbox.")
selected_resi = str(args.mol_index)
if selected_resi not in resis:
    raise ValueError(
        f"Molecule/residue {selected_resi!r} is not in the selected subbox. "
        f"Available molecules/residues: {resis}"
    )
selected_pos = resis.index(selected_resi)

print("GRO title:", frame["title"])
print("cell A:", a, b, c)
print("subbox A:", xmin, xmax, ymin, ymax, zmin, zmax)
print("molecules/residues in subbox:", resis)
print("selected molecule/residue:", selected_resi)

for i, resi in enumerate(resis):
    if resi == selected_resi:
        continue

    obj_name = f"mol_{i}_resi_{resi}"
    color_name = f"mol_color_{i}"
    cmd.set_color(color_name, palette[i % len(palette)])
    cmd.create(obj_name, f"subbox_obj and resi {resi}")
    cmd.hide("everything", obj_name)

    should_show_surface = (
        args.surface_mode == "all"
    )
    if should_show_surface:
        cmd.show("surface", obj_name)
        cmd.set("surface_mode", 1, obj_name)
        cmd.set("solvent_radius", 0.01, obj_name)
        cmd.set("surface_quality", 2, obj_name)
        cmd.set("surface_color", color_name, obj_name)
        cmd.set("transparency", 0.9, obj_name)

selected_obj_name = f"mol_{selected_pos}_resi_{selected_resi}"
selected_color_name = "selected_burgundy"
cmd.set_color(selected_color_name, [0.45, 0.02, 0.12])
cmd.create(selected_obj_name, f"subbox_obj and resi {selected_resi}")
cmd.hide("everything", selected_obj_name)

if args.surface_mode != "none":
    cmd.show("surface", selected_obj_name)
    cmd.set("surface_mode", 1, selected_obj_name)
    cmd.set("solvent_radius", 0.01, selected_obj_name)
    cmd.set("surface_quality", 2, selected_obj_name)
    cmd.set("surface_color", selected_color_name, selected_obj_name)
    cmd.set("transparency", 0.0, selected_obj_name)

cmd.disable("subbox_obj")

cmd.rebuild()

cmd.enable("subbox_obj")
cmd.center("subbox_obj")
cmd.zoom("subbox_obj", 10)
cmd.disable("subbox_obj")

if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(out), width=args.width, height=args.height, dpi=300, ray=int(args.ray))
python end
