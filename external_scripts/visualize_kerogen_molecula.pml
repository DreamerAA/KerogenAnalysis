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
        description="Render one kerogen molecule/residue from a PDB file."
    )
    parser.add_argument("--ker-pdb", default="ker.pdb", help="Kerogen PDB file")
    parser.add_argument("--chain", default="A", help="PDB chain to render")
    parser.add_argument("--resi", "--resid", default="1", dest="resi", help="Residue number")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG output path.",
    )
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ray", dest="ray", action="store_true", help="Use ray tracing for PNG output")
    parser.add_argument("--no-ray", dest="ray", action="store_false", help="Use the OpenGL framebuffer for PNG output")
    parser.set_defaults(ray=False)
    parser.add_argument(
        "--camera-rot",
        nargs=3,
        type=float,
        default=(90.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Camera rotations in degrees after orienting the molecule.",
    )
    parser.add_argument("--zoom-buffer", type=float, default=5.0)
    args, _unknown = parser.parse_known_args(script_args)
    return args


args = parse_args()

cmd.load(args.ker_pdb, "kerogen")

selection = f"chain {args.chain} and resi {args.resi}"
cmd.select("ker1", selection)
if cmd.count_atoms("ker1") == 0:
    raise ValueError(f"No atoms selected by {selection!r} in {args.ker_pdb}")

cmd.create("ker1_obj", "ker1")

cmd.disable("kerogen")
cmd.enable("ker1_obj")

cmd.hide("everything")
cmd.set("orthoscopic", "on")

cmd.orient("ker1_obj")
for axis, angle in zip(("x", "y", "z"), args.camera_rot):
    if angle:
        cmd.turn(axis, angle)

cmd.show("sticks", "ker1_obj")
cmd.set("stick_radius", 0.06, "ker1_obj")

cmd.show("spheres", "ker1_obj")
cmd.set("sphere_scale", 0.35, "ker1_obj")

cmd.color("gray70", "ker1_obj and elem C")
cmd.color("white", "ker1_obj and elem H")
cmd.color("red", "ker1_obj and elem O")
cmd.color("blue", "ker1_obj and elem N")
cmd.color("yellow", "ker1_obj and elem S")

cmd.bg_color("white")
cmd.set("ray_opaque_background", "off")
cmd.set("antialias", 2)
cmd.set("valence", 0)

cmd.orient("ker1_obj")
cmd.zoom("ker1_obj", args.zoom_buffer)

if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(out), width=args.width, height=args.height, dpi=args.dpi, ray=int(args.ray))
python end
