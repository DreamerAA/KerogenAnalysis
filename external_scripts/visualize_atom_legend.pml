# visualize_atom_legend.pml
# Standalone PyMOL script: renders separate atom spheres with the same colors
# as in visualize_kerogen_molecula.pml.
#
# Output:
#   ./figs/atom_legend_labeled.png      -- spheres + labels
#   ./figs/atom_legend_spheres_only.png -- only colored spheres

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

    parser = argparse.ArgumentParser(description="Render atom color legend.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG output path for the labeled legend.",
    )
    parser.add_argument(
        "--output-spheres",
        default=None,
        help=(
            "Optional PNG output path for the spheres-only legend. "
            "If omitted and --output is set, '<output stem>_spheres_only.png' is used."
        ),
    )
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=650)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ray", dest="ray", action="store_true", help="Use ray tracing for PNG output")
    parser.add_argument("--no-ray", dest="ray", action="store_false", help="Use the OpenGL framebuffer for PNG output")
    parser.set_defaults(ray=False)
    args, _unknown = parser.parse_known_args(script_args)
    return args


args = parse_args()
if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output_spheres is None:
        out = Path(args.output)
        args.output_spheres = str(out.with_name(f"{out.stem}_spheres_only{out.suffix}"))
if args.output_spheres:
    Path(args.output_spheres).parent.mkdir(parents=True, exist_ok=True)
python end

# -----------------------------------------------------------------------------
# Scene settings
# -----------------------------------------------------------------------------
bg_color white
set orthoscopic, on
set ray_opaque_background, off
set antialias, 2
set valence, 0
set two_sided_lighting, on
set specular, 0.35
set shininess, 35
set ambient, 0.25
set direct, 0.65

# -----------------------------------------------------------------------------
# Atom samples / legend entries
# -----------------------------------------------------------------------------
# The objects are pseudoatoms, so this script does not require a PDB file.
# Positions are arranged horizontally for convenient use as a subfigure.

pseudoatom atom_C, pos=[-4.8, 0.0, 0.0], elem=C, name=C, vdw=1.0
pseudoatom atom_H, pos=[-2.4, 0.0, 0.0], elem=H, name=H, vdw=1.0
pseudoatom atom_O, pos=[ 0.0, 0.0, 0.0], elem=O, name=O, vdw=1.0
pseudoatom atom_N, pos=[ 2.4, 0.0, 0.0], elem=N, name=N, vdw=1.0
pseudoatom atom_S, pos=[ 4.8, 0.0, 0.0], elem=S, name=S, vdw=1.0

hide everything
show spheres, atom_C or atom_H or atom_O or atom_N or atom_S
set sphere_scale, 0.55, atom_C or atom_H or atom_O or atom_N or atom_S

# Same color mapping as in the molecular figure
color gray70, atom_C
color white,  atom_H
color red,    atom_O
color blue,   atom_N
color yellow, atom_S

# Make the white hydrogen visible on white/transparent backgrounds.
# If you want pure white H without a visible rim, comment this block out.
set ray_trace_mode, 1
set ray_trace_color, black
set ray_trace_gain, 0.05

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
# You can change labels below, e.g. "Carbon (C)" instead of "C".
label atom_C, "C"
label atom_H, "H"
label atom_O, "O"
label atom_N, "N"
label atom_S, "S"

set label_color, black
set label_size, 36
set label_font_id, 7
set label_outline_color, white
set label_position, [0.0, -1.15, 0.0], atom_C or atom_H or atom_O or atom_N or atom_S

# -----------------------------------------------------------------------------
# Camera and rendering
# -----------------------------------------------------------------------------
orient atom_C or atom_H or atom_O or atom_N or atom_S
zoom atom_C or atom_H or atom_O or atom_N or atom_S, 2.0

python
if args.output:
    cmd.png(str(args.output), width=args.width, height=args.height, dpi=args.dpi, ray=int(args.ray))
python end

# Also save a clean version without text labels, in case labels are added later
hide labels
python
if args.output_spheres:
    cmd.png(str(args.output_spheres), width=args.width, height=args.height, dpi=args.dpi, ray=int(args.ray))
python end
