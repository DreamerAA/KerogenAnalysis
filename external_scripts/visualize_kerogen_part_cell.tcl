package require pbctools

mol delete all

# Загружаем только первый кадр
mol new ./ker.pdb type pdb first 0 last 0 step 1 waitfor all

# =========================
# Settings
# =========================

set molid [molinfo top]


set a [molinfo $molid get a]
set b [molinfo $molid get b]
set c [molinfo $molid get c]

puts "Cell size:"
puts "a = $a"
puts "b = $b"
puts "c = $c"


# Границы малой ячейки
set box_size 30.0

set xmin [expr {0.5*$a - 0.5*$box_size}]
set xmax [expr {0.5*$a + 0.5*$box_size}]

set ymin [expr {0.5*$b - 0.5*$box_size}]
set ymax [expr {0.5*$b + 0.5*$box_size}]

set zmin [expr {0.5*$c - 0.5*$box_size}]
set zmax [expr {0.5*$c + 0.5*$box_size}]

puts "Sub-box:"
puts "x = $xmin ... $xmax"
puts "y = $ymin ... $ymax"
puts "z = $zmin ... $zmax"


set keep_sel "x >= $xmin and x <= $xmax and y >= $ymin and y <= $ymax and z >= $zmin and z <= $zmax"

set keep [atomselect $molid $keep_sel]

# Создаём новую молекулу только из атомов внутри box
$keep writepdb subbox.pdb

mol delete $molid
mol new subbox.pdb type pdb waitfor all

set molid [molinfo top]


# =========================
# Drawing procedures
# =========================

proc draw_box {molid xmin xmax ymin ymax zmin zmax} {
 graphics $molid color black
    graphics $molid materials off

    set p000 [list $xmin $ymin $zmin]
    set p100 [list $xmax $ymin $zmin]
    set p010 [list $xmin $ymax $zmin]
    set p110 [list $xmax $ymax $zmin]
    set p001 [list $xmin $ymin $zmax]
    set p101 [list $xmax $ymin $zmax]
    set p011 [list $xmin $ymax $zmax]
    set p111 [list $xmax $ymax $zmax]

    graphics $molid line $p000 $p100 width 4
    graphics $molid line $p000 $p010 width 4
    graphics $molid line $p100 $p110 width 4
    graphics $molid line $p010 $p110 width 4

    graphics $molid line $p001 $p101 width 4
    graphics $molid line $p001 $p011 width 4
    graphics $molid line $p101 $p111 width 4
    graphics $molid line $p011 $p111 width 4

    graphics $molid line $p000 $p001 width 4
    graphics $molid line $p100 $p101 width 4
    graphics $molid line $p010 $p011 width 4
    graphics $molid line $p110 $p111 width 4
}

proc draw_dark_box_faces {molid xmin xmax ymin ymax zmin zmax} {
    graphics $molid color black
    graphics $molid material DarkFog

    set p000 [list $xmin $ymin $zmin]
    set p100 [list $xmax $ymin $zmin]
    set p010 [list $xmin $ymax $zmin]
    set p110 [list $xmax $ymax $zmin]
    set p001 [list $xmin $ymin $zmax]
    set p101 [list $xmax $ymin $zmax]
    set p011 [list $xmin $ymax $zmax]
    set p111 [list $xmax $ymax $zmax]

    graphics $molid triangle $p000 $p100 $p110
    graphics $molid triangle $p000 $p110 $p010

    graphics $molid triangle $p001 $p101 $p111
    graphics $molid triangle $p001 $p111 $p011

    graphics $molid triangle $p000 $p100 $p101
    graphics $molid triangle $p000 $p101 $p001

    graphics $molid triangle $p010 $p110 $p111
    graphics $molid triangle $p010 $p111 $p011

    graphics $molid triangle $p000 $p010 $p011
    graphics $molid triangle $p000 $p011 $p001

    graphics $molid triangle $p100 $p110 $p111
    graphics $molid triangle $p100 $p111 $p101
}

# =========================
# Select full molecules inside sub-box
# =========================

set boxsel "x >= $xmin and x <= $xmax and y >= $ymin and y <= $ymax and z >= $zmin and z <= $zmax"

set candidates [atomselect $molid $boxsel]
set candidate_residues [lsort -unique [$candidates get residue]]

set selected_residues {}

foreach r $candidate_residues {
    set mol_sel [atomselect $molid "residue $r"]

    set xs [$mol_sel get x]
    set ys [$mol_sel get y]
    set zs [$mol_sel get z]

    set ok 1

    foreach x $xs y $ys z $zs {
        if {$x < $xmin || $x > $xmax || $y < $ymin || $y > $ymax || $z < $zmin || $z > $zmax} {
            set ok 0
            break
        }
    }

    if {$ok == 1} {
        lappend selected_residues $r
    }

    $mol_sel delete
}

set candidates [atomselect $molid $boxsel]
set selected_residues [lsort -unique [$candidates get residue]]
$candidates delete

set reslist [join $selected_residues " "]

puts "Molecules intersecting box: [llength $selected_residues]"
puts "Residues: $reslist"

# =========================
# Clear old representations
# =========================

set nreps [molinfo $molid get numreps]
for {set i [expr {$nreps - 1}]} {$i >= 0} {incr i -1} {
    mol delrep $i $molid
}

graphics $molid delete all

# =========================
# Materials
# =========================

material add DarkFog
material change opacity DarkFog 0.12
material change ambient DarkFog 0.00
material change diffuse DarkFog 0.05
material change specular DarkFog 0.00

material add MolSurface
material change opacity MolSurface 0.70
#material change ambient MolSurface 0.25
#material change diffuse MolSurface 0.70
#material change specular MolSurface 0.05

# =========================
# Draw dark volume first
# =========================

# draw_dark_box_faces $molid $xmin $xmax $ymin $ymax $zmin $zmax

# =========================
# Draw molecule surface
# =========================

if {[llength $selected_residues] > 0} {
    color change rgb 30 0.62 0.53 0.22

    # mol representation QuickSurf 0.4 0.10 0.6 5
    mol representation Surf 0.85 0.0
    # mol representation MSMS 1.2 3.0
    mol color ColorID 30
    mol selection "residue $reslist"
    mol material MolSurface
    mol addrep $molid

    # =========================
    # Draw atoms and bonds
    # =========================

    mol representation CPK 0.7 0.3 8 8

    mol color Element
    mol selection "residue $reslist"
    mol material AOShiny
    mol addrep $molid
}

# =========================
# Draw box last
# =========================

draw_box $molid $xmin $xmax $ymin $ymax $zmin $zmax

# =========================
# View settings
# =========================

display background white
axes location Off