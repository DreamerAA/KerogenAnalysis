package require pbctools

proc strip_outer_quotes {value} {
    if {[regexp {^'(.*)'$} $value -> inner] || [regexp {^"(.*)"$} $value -> inner]} {
        return $inner
    }
    return $value
}

proc script_args {} {
    global argv env
    if {[llength $argv] > 0} {
        return $argv
    }
    if {[info exists env(VMD_SCRIPT_ARGS)] && $env(VMD_SCRIPT_ARGS) ne ""} {
        set raw [split $env(VMD_SCRIPT_ARGS)]
        set result {}
        foreach item $raw {
            lappend result [strip_outer_quotes $item]
        }
        return $result
    }
    return {}
}

proc parse_args {} {
    array set opts {
        --ker-pdb ker.pdb
        --box-size 30.0
        --center {}
        --xmin {}
        --xmax {}
        --ymin {}
        --ymax {}
        --zmin {}
        --zmax {}
        --subbox-pdb subbox.pdb
        --rotate-x 0
        --rotate-y 0
        --rotate-z 0
        --scale 1.0
        --output {}
    }

    set args [script_args]
    set i 0
    while {$i < [llength $args]} {
        set key [lindex $args $i]
        if {$key eq "--"} {
            incr i
            continue
        }
        if {![string match --* $key]} {
            incr i
            continue
        }
        if {$key ni [array names opts]} {
            incr i
            if {$i < [llength $args] && ![string match --* [lindex $args $i]]} {
                incr i
            }
            continue
        }
        incr i
        if {$i >= [llength $args]} {
            error "Missing value for $key"
        }

        if {$key eq "--center"} {
            if {$i + 2 >= [llength $args]} {
                error "Missing values for --center X Y Z"
            }
            set opts($key) [list [lindex $args $i] [lindex $args [expr {$i + 1}]] [lindex $args [expr {$i + 2}]]]
            incr i 3
        } else {
            set opts($key) [strip_outer_quotes [lindex $args $i]]
            incr i
        }
    }
    return [array get opts]
}

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

array set opts [parse_args]

mol delete all
mol new $opts(--ker-pdb) type pdb first 0 last 0 step 1 waitfor all

set molid [molinfo top]

set a [molinfo $molid get a]
set b [molinfo $molid get b]
set c [molinfo $molid get c]

puts "Cell size:"
puts "a = $a"
puts "b = $b"
puts "c = $c"

if {$opts(--xmin) ne "" && $opts(--xmax) ne "" && $opts(--ymin) ne "" && $opts(--ymax) ne "" && $opts(--zmin) ne "" && $opts(--zmax) ne ""} {
    set xmin $opts(--xmin)
    set xmax $opts(--xmax)
    set ymin $opts(--ymin)
    set ymax $opts(--ymax)
    set zmin $opts(--zmin)
    set zmax $opts(--zmax)
} else {
    if {$opts(--center) eq ""} {
        set cx [expr {0.5 * $a}]
        set cy [expr {0.5 * $b}]
        set cz [expr {0.5 * $c}]
    } else {
        set cx [lindex $opts(--center) 0]
        set cy [lindex $opts(--center) 1]
        set cz [lindex $opts(--center) 2]
    }
    set half [expr {0.5 * $opts(--box-size)}]
    set xmin [expr {$cx - $half}]
    set xmax [expr {$cx + $half}]
    set ymin [expr {$cy - $half}]
    set ymax [expr {$cy + $half}]
    set zmin [expr {$cz - $half}]
    set zmax [expr {$cz + $half}]
}

puts "Sub-box:"
puts "x = $xmin ... $xmax"
puts "y = $ymin ... $ymax"
puts "z = $zmin ... $zmax"

set keep_sel "x >= $xmin and x <= $xmax and y >= $ymin and y <= $ymax and z >= $zmin and z <= $zmax"
set keep [atomselect $molid $keep_sel]
$keep writepdb $opts(--subbox-pdb)
$keep delete

mol delete $molid
mol new $opts(--subbox-pdb) type pdb waitfor all

set molid [molinfo top]
set boxsel "x >= $xmin and x <= $xmax and y >= $ymin and y <= $ymax and z >= $zmin and z <= $zmax"

set candidates [atomselect $molid $boxsel]
set selected_residues [lsort -unique [$candidates get residue]]
$candidates delete

set reslist [join $selected_residues " "]

puts "Molecules intersecting box: [llength $selected_residues]"
puts "Residues: $reslist"

set nreps [molinfo $molid get numreps]
for {set i [expr {$nreps - 1}]} {$i >= 0} {incr i -1} {
    mol delrep $i $molid
}

graphics $molid delete all

material add DarkFog
material change opacity DarkFog 0.12
material change ambient DarkFog 0.00
material change diffuse DarkFog 0.05
material change specular DarkFog 0.00

material add MolSurface
material change opacity MolSurface 0.70

if {[llength $selected_residues] > 0} {
    color change rgb 30 0.62 0.53 0.22

    mol representation Surf 0.85 0.0
    mol color ColorID 30
    mol selection "residue $reslist"
    mol material MolSurface
    mol addrep $molid

    mol representation CPK 0.7 0.3 8 8
    mol color Element
    mol selection "residue $reslist"
    mol material AOShiny
    mol addrep $molid
}

draw_box $molid $xmin $xmax $ymin $ymax $zmin $zmax

display background white
axes location Off
if {$opts(--rotate-x) != 0} { rotate x by $opts(--rotate-x) }
if {$opts(--rotate-y) != 0} { rotate y by $opts(--rotate-y) }
if {$opts(--rotate-z) != 0} { rotate z by $opts(--rotate-z) }
scale by $opts(--scale)

if {$opts(--output) ne ""} {
    set out_dir [file dirname $opts(--output)]
    if {$out_dir ne "."} {
        file mkdir $out_dir
    }
    render snapshot $opts(--output)
}
