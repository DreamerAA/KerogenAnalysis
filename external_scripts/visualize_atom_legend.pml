bash -lc cat > /mnt/data/visualize_atom_legend.pml <<'EOF'
# visualize_atom_legend.pml
# Standalone PyMOL script: renders separate atom spheres with the same colors
# as in visualize_kerogen_molecula.pml.
#
# Output:
#   ./figs/atom_legend_labeled.png      -- spheres + labels
#   ./figs/atom_legend_spheres_only.png -- only colored spheres

reinitialize

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

# Create output directory if it does not exist
python
import os
os.makedirs('./figs', exist_ok=True)
python end

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

png ./figs/atom_legend_labeled.png, width=2200, height=650, dpi=300, ray=1

# Also save a clean version without text labels, in case labels are added later
hide labels
png ./figs/atom_legend_spheres_only.png, width=2200, height=650, dpi=300, ray=1
EOF
sed -n '1,240p' /mnt/data/visualize_atom_legend.pml