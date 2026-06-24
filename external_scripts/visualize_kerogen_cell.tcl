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
        --cell {62.309 74.106 130.150 90 90 90}
        --periodic-axes xyz
        --periodic-count 1
        --rotate-x 0
        --rotate-y 110
        --rotate-z 0
        --scale 1.6
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
        set opts($key) [strip_outer_quotes [lindex $args $i]]
        incr i
    }
    return [array get opts]
}

array set opts [parse_args]

mol delete all
mol new $opts(--ker-pdb) type pdb first 0 last 0 step 1 waitfor all

set molid [molinfo top]

pbc set $opts(--cell) -molid $molid

puts "PBC cell:"
puts [pbc get -molid $molid]

pbc box -molid $molid -color black

foreach axis [split $opts(--periodic-axes) ""] {
    if {$axis in {x y z}} {
        mol showperiodic $molid 0 $axis
    }
}
mol numperiodic $molid 0 $opts(--periodic-count)

set sel [atomselect $molid all]
set fragments [lsort -integer -unique [$sel get fragment]]

puts "Atoms: [$sel num]"
puts "Fragments: [llength $fragments]"

set nreps [molinfo $molid get numreps]
for {set r [expr {$nreps - 1}]} {$r >= 0} {incr r -1} {
    mol delrep $r $molid
}

set i 0
foreach frag $fragments {
    mol addrep $molid
    mol modselect $i $molid "fragment $frag"
    mol modstyle $i $molid CPK 0.7 0.3 8 8
    mol modcolor $i $molid ColorID [expr {$i % 32}]
    incr i
}

display projection Perspective
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

$sel delete
