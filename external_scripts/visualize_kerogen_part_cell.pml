reinitialize

python
import argparse
import os
import re
from pathlib import Path
import shlex
import sys

from pymol import cmd


def parse_args():
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not script_args:
        script_args = shlex.split(os.environ.get("PYMOL_SCRIPT_ARGS", ""))

    parser = argparse.ArgumentParser(
        description="Render a small subbox from a kerogen PDB cell."
    )
    parser.add_argument("--ker-pdb", default="ker.pdb", help="Kerogen PDB file")
    parser.add_argument("--sim-gro", default=None, help="Optional GRO trajectory; coordinates are taken from the selected frame")
    parser.add_argument("--frame", type=int, default=0, help="0-based frame index in GRO (ignored when --step or --time are given)")
    parser.add_argument("--step", type=int, default=None, help="MD step number from GRO title, e.g. 25000 from 'step= 25000'")
    parser.add_argument("--time", dest="time_ps", type=float, default=None, help="Simulation time in ps from GRO title, e.g. 50.0 from 't= 50.00000'")
    parser.add_argument("--resname", default="KRG", help="Residue name to extract from GRO")
    parser.add_argument("--box-size", type=float, default=20.0, help="Subbox side in Angstrom")
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Subbox center in Angstrom. Defaults to the PDB cell center.",
    )
    parser.add_argument("--xmin", type=float, default=None, help="Explicit subbox X min in Angstrom")
    parser.add_argument("--xmax", type=float, default=None, help="Explicit subbox X max in Angstrom")
    parser.add_argument("--ymin", type=float, default=None, help="Explicit subbox Y min in Angstrom")
    parser.add_argument("--ymax", type=float, default=None, help="Explicit subbox Y max in Angstrom")
    parser.add_argument("--zmin", type=float, default=None, help="Explicit subbox Z min in Angstrom")
    parser.add_argument("--zmax", type=float, default=None, help="Explicit subbox Z max in Angstrom")
    parser.add_argument("--output", default=None, help="Optional PNG output path")
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ray", dest="ray", action="store_true", help="Use ray tracing for PNG output")
    parser.add_argument("--no-ray", dest="ray", action="store_false", help="Use the OpenGL framebuffer for PNG output")
    parser.set_defaults(ray=False)
    parser.add_argument(
        "--surface-mode",
        choices=("all", "none"),
        default="all",
        help="Whether to render molecule surfaces.",
    )
    parser.add_argument("--surface-transparency", type=float, default=0.4)
    parser.add_argument(
        "--camera-rot",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Camera rotations in degrees after orienting the subbox.",
    )
    parser.add_argument("--zoom-buffer", type=float, default=3.0)
    args, _unknown = parser.parse_known_args(script_args)
    return args


def read_gro_frame(path, resname, max_atoms, frame_index=None, step=None, time_ps=None):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        current_frame = 0
        while True:
            title = f.readline()
            if not title:
                break
            natoms_line = f.readline()
            if not natoms_line:
                raise ValueError(f"Unexpected EOF in {path}")
            natoms = int(natoms_line.strip())

            if step is not None:
                m = re.search(r'step=\s*(\d+)', title)
                frame_matches = m is not None and int(m.group(1)) == step
            elif time_ps is not None:
                m = re.search(r't=\s*([\d.]+)', title)
                frame_matches = m is not None and abs(float(m.group(1)) - time_ps) < 1e-3
            else:
                frame_matches = (current_frame == frame_index)

            coords = []
            matched_atoms = 0
            for _ in range(natoms):
                line = f.readline()
                if not frame_matches:
                    continue
                line_resname = line[5:10].strip()
                if resname != "ALL" and line_resname != resname:
                    continue
                matched_atoms += 1
                if len(coords) < max_atoms:
                    coords.append((
                        float(line[20:28]) * 10.0,
                        float(line[28:36]) * 10.0,
                        float(line[36:44]) * 10.0,
                    ))

            box_line = f.readline()
            if not box_line:
                raise ValueError(f"Unexpected EOF after atoms in {path}")

            if frame_matches:
                box_values = [float(v) * 10.0 for v in box_line.split()]
                if len(box_values) < 3:
                    raise ValueError(f"Cannot parse GRO box line: {box_line!r}")
                return {
                    "title": title.strip(),
                    "coords": coords,
                    "matched_atoms": matched_atoms,
                    "box": tuple(box_values[:3]),
                }
            current_frame += 1

    if step is not None:
        raise ValueError(f"Step {step} not found in {path}")
    elif time_ps is not None:
        raise ValueError(f"Time {time_ps} ps not found in {path}")
    else:
        raise ValueError(f"Frame {frame_index} not found in {path}")


args = parse_args()

cmd.load(args.ker_pdb, "kerogen")

if args.sim_gro:
    template_atoms = cmd.get_model("kerogen").atom
    template_count = len(template_atoms)

    frame = read_gro_frame(
        args.sim_gro, args.resname, template_count,
        frame_index=args.frame if (args.step is None and args.time_ps is None) else None,
        step=args.step,
        time_ps=args.time_ps,
    )
    gro_coords = frame["coords"]

    if len(gro_coords) < template_count:
        raise ValueError(
            f"Not enough GRO atoms: got {len(gro_coords)}, need {template_count}. "
            f"Check --resname {args.resname!r}."
        )

    coord_by_index = {atom.index: coord for atom, coord in zip(template_atoms, gro_coords)}
    cmd.alter_state(1, "kerogen", "(x, y, z) = coord_by_index[index]", space=locals())
    cmd.set_symmetry("kerogen", frame["box"][0], frame["box"][1], frame["box"][2], 90.0, 90.0, 90.0, "P 1")
    print("GRO frame:", frame["title"])
    print("GRO box (Å):", frame["box"])

cmd.hide("everything")
cmd.bg_color("white")
cmd.set("orthoscopic", "off")
cmd.set("antialias", 2)
cmd.set("ray_opaque_background", "off")
cmd.set("specular", 0.2)
cmd.set("shininess", 10)

cell = cmd.get_symmetry("kerogen")
a, b, c = cell[0], cell[1], cell[2]
if not all((a, b, c)):
    raise ValueError(f"No CRYST1/cell dimensions found in {args.ker_pdb}")

if all(v is not None for v in (args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax)):
    xmin, xmax = args.xmin, args.xmax
    ymin, ymax = args.ymin, args.ymax
    zmin, zmax = args.zmin, args.zmax
else:
    if args.center is None:
        cx, cy, cz = 0.5 * a, 0.5 * b, 0.5 * c
    else:
        cx, cy, cz = args.center
    half = 0.5 * args.box_size
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half
    zmin, zmax = cz - half, cz + half

cmd.select(
    "sub_atoms",
    f"x>{xmin} and x<{xmax} and y>{ymin} and y<{ymax} and z>{zmin} and z<{zmax}",
)
if cmd.count_atoms("sub_atoms") == 0:
    raise ValueError(
        "Selected subbox is empty. Try a larger --box-size or pass --center X Y Z."
    )

cmd.select("sub_mols", "sub_atoms")

print("cell:", a, b, c)
print("subbox:", xmin, xmax, ymin, ymax, zmin, zmax)

cmd.create("subbox_obj", "sub_mols")
cmd.disable("kerogen")
cmd.enable("subbox_obj")

cmd.hide("everything")
cmd.show("sticks", "subbox_obj")
cmd.show("spheres", "subbox_obj")
cmd.color("gray70", "subbox_obj and elem C")
cmd.color("white",  "subbox_obj and elem H")
cmd.color("red",    "subbox_obj and elem O")
cmd.color("blue",   "subbox_obj and elem N")
cmd.color("yellow", "subbox_obj and elem S")
cmd.set("sphere_scale", 0.22, "subbox_obj")
cmd.set("stick_radius", 0.1, "subbox_obj")
cmd.hide("surface", "subbox_obj")

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

if args.surface_mode == "all":
    for i, resi in enumerate(resis):
        color_name = f"mol_surface_color_{i}"
        obj_name = f"mol_surface_{resi}"

        cmd.set_color(color_name, palette[i % len(palette)])
        cmd.create(obj_name, f"subbox_obj and resi {resi}")
        cmd.hide("everything", obj_name)
        cmd.set("surface_mode", 1, obj_name)
        cmd.set("solvent_radius", 0.01, obj_name)
        cmd.set("surface_quality", 2, obj_name)
        cmd.set("surface_color", color_name, obj_name)
        cmd.set("transparency", args.surface_transparency, obj_name)
        cmd.show("surface", obj_name)

cmd.hide("surface", "subbox_obj")
cmd.show("sticks", "subbox_obj")
cmd.show("spheres", "subbox_obj")
cmd.color("gray70", "subbox_obj and elem C")
cmd.color("white",  "subbox_obj and elem H")
cmd.color("red",    "subbox_obj and elem O")
cmd.color("blue",   "subbox_obj and elem N")
cmd.color("yellow", "subbox_obj and elem S")

cmd.rebuild()

from pymol.cgo import BEGIN, END, LINES, COLOR, VERTEX, LINEWIDTH
box_cgo = [
    LINEWIDTH, 2.5,
    BEGIN, LINES,
    COLOR, 0.0, 0.0, 0.0,
    VERTEX, xmin, ymin, zmin, VERTEX, xmax, ymin, zmin,
    VERTEX, xmin, ymax, zmin, VERTEX, xmax, ymax, zmin,
    VERTEX, xmin, ymin, zmax, VERTEX, xmax, ymin, zmax,
    VERTEX, xmin, ymax, zmax, VERTEX, xmax, ymax, zmax,
    VERTEX, xmin, ymin, zmin, VERTEX, xmin, ymax, zmin,
    VERTEX, xmax, ymin, zmin, VERTEX, xmax, ymax, zmin,
    VERTEX, xmin, ymin, zmax, VERTEX, xmin, ymax, zmax,
    VERTEX, xmax, ymin, zmax, VERTEX, xmax, ymax, zmax,
    VERTEX, xmin, ymin, zmin, VERTEX, xmin, ymin, zmax,
    VERTEX, xmax, ymin, zmin, VERTEX, xmax, ymin, zmax,
    VERTEX, xmin, ymax, zmin, VERTEX, xmin, ymax, zmax,
    VERTEX, xmax, ymax, zmin, VERTEX, xmax, ymax, zmax,
    END,
]
cmd.load_cgo(box_cgo, "subbox_outline")

cmd.orient("subbox_obj")
for axis, angle in zip(("x", "y", "z"), args.camera_rot):
    if angle:
        cmd.turn(axis, angle)
cmd.zoom("subbox_obj", args.zoom_buffer)

if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(out), width=args.width, height=args.height, dpi=args.dpi, ray=int(args.ray))
python end
