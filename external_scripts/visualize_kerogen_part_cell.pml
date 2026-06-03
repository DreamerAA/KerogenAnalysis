reinitialize
load ker.pdb, kerogen

hide everything
bg_color white
set orthoscopic, off
set antialias, 2
set ray_opaque_background, off

# чуть лучше свет
set specular, 0.2
set shininess, 10


# границы малой ячейки
python
from pymol import cmd
box_size = 20.0

# если ячейка из CRYST1 прочиталась
cell = cmd.get_symmetry("kerogen")
a, b, c = cell[0], cell[1], cell[2]

xmin, xmax = 0.5*a - 0.5*box_size, 0.5*a + 0.5*box_size
ymin, ymax = 0.5*b - 0.5*box_size, 0.5*b + 0.5*box_size
zmin, zmax = 0.5*c - 0.5*box_size, 0.5*c + 0.5*box_size

cmd.select("sub_atoms", f"x>{xmin} and x<{xmax} and y>{ymin} and y<{ymax} and z>{zmin} and z<{zmax}")
cmd.select("sub_mols", "sub_atoms")
# cmd.select("sub_mols", "byres sub_atoms")

print("cell:", a, b, c)
print("subbox:", xmin, xmax, ymin, ymax, zmin, zmax)
python end

create subbox_obj, sub_mols
disable kerogen
enable subbox_obj


# создать цвет
set_color gray, [0.6, 0.6, 0.6]
set_color dirtyyellow, [0.62, 0.53, 0.22]


# показываем только атомы внутри выбранного объекта
hide everything
show sticks, subbox_obj
show spheres, subbox_obj


# атомы и связи одним цветом
color gray, subbox_obj

# атомы и связи
set sphere_scale, 0.22, subbox_obj
set stick_radius, 0.1, subbox_obj

# поверхность ближе/дальше от атомов
# set surface_mode, 0, subbox_obj
# set surface_quality, 2, subbox_obj
# set solvent_radius, 0.1, subbox_obj


# скрываем общую поверхность, оставляем атомы/связи общими
hide surface, subbox_obj

python
from pymol import cmd

# палитра для разных молекул
palette = [
    [0.62, 0.53, 0.22],  # dirty yellow
    [0.45, 0.58, 0.35],  # olive green
    [0.55, 0.42, 0.30],  # brown
    [0.45, 0.50, 0.62],  # muted blue
    [0.62, 0.42, 0.42],  # muted red
    [0.50, 0.45, 0.62],  # violet
]

model = cmd.get_model("subbox_obj")
resis = sorted(set(atom.resi for atom in model.atom), key=lambda x: int(x) if x.isdigit() else x)

for i, resi in enumerate(resis):
    color_name = f"mol_surface_color_{i}"
    obj_name = f"mol_surface_{resi}"

    rgb = palette[i % len(palette)]
    cmd.set_color(color_name, rgb)

    cmd.create(obj_name, f"subbox_obj and resi {resi}")

    cmd.hide("everything", obj_name)

    # cmd.alter(obj_name, "vdw = vdw * 0.5")

    # cmd.set("surface_mode", 0, obj_name)
    # cmd.set("solvent_radius", 0.0, obj_name)
    
    cmd.set("surface_mode", 1, obj_name)
    cmd.set("solvent_radius", 0.01, obj_name)

    cmd.set("surface_quality", 2, obj_name)
    cmd.set("surface_color", color_name, obj_name)
    cmd.set("transparency", 0.4, obj_name)

    cmd.show("surface", obj_name)

# На всякий случай запрещаем общую поверхность у исходного объекта
cmd.hide("surface", "subbox_obj")
cmd.show("sticks", "subbox_obj")
cmd.show("spheres", "subbox_obj")
cmd.color("gray", "subbox_obj")

cmd.rebuild()
python end


# # рамка малой ячейки
# python
# from pymol import cmd
# obj = "subbox_frame"
# cmd.delete(obj)
# 
# pts = {
# "000": (xmin,ymin,zmin), "100": (xmax,ymin,zmin),
# "010": (xmin,ymax,zmin), "110": (xmax,ymax,zmin),
# "001": (xmin,ymin,zmax), "101": (xmax,ymin,zmax),
# "011": (xmin,ymax,zmax), "111": (xmax,ymax,zmax),
# }
# 
# edges = [
# ("000","100"), ("000","010"), ("100","110"), ("010","110"),
# ("001","101"), ("001","011"), ("101","111"), ("011","111"),
# ("000","001"), ("100","101"), ("010","011"), ("110","111")
# ]
# 
# from pymol.cgo import BEGIN, LINES, VERTEX, END, COLOR, LINEWIDTH
# cgo = [LINEWIDTH, 8.0, COLOR, 0.0, 0.0, 0.0, BEGIN, LINES]
# for a1, a2 in edges:
#     cgo += [VERTEX, *pts[a1], VERTEX, *pts[a2]]
# cgo += [END]
# cmd.load_cgo(cgo, obj)
# python end

zoom subbox_obj, 3