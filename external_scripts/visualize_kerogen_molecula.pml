load ./ker.pdb, kerogen

select ker1, chain A and resi 1
create ker1_obj, ker1

disable kerogen
enable ker1_obj

hide everything


set orthoscopic, on

# Повернуть молекулу так, чтобы смотреть почти вдоль одной оси
orient ker1_obj
turn x, 90
turn z, 0


# === Bonds ===
show sticks, ker1_obj
set stick_radius, 0.06, ker1_obj

# === Atoms ===
show spheres, ker1_obj
set sphere_scale, 0.35, ker1_obj

# === Colors ===
color gray70, ker1_obj and elem C
color white,  ker1_obj and elem H
color red,    ker1_obj and elem O
color blue,   ker1_obj and elem N
color yellow, ker1_obj and elem S

bg_color white
set orthoscopic, on
set ray_opaque_background, off
set antialias, 2
set valence, 0

orient ker1_obj
zoom ker1_obj, 5

png ./figs/kerogen_chainA_res1.png, width=2200, height=1800, dpi=300, ray=1