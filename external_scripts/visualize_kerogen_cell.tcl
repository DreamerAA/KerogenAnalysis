package require pbctools

mol delete all

# Загружаем только первый кадр
mol new ./ker.pdb type pdb first 0 last 0 step 1 waitfor all

set molid [molinfo top]

# Явно задаём ячейку
pbc set {62.309 74.106 130.150 90 90 90} -molid $molid

puts "PBC cell:"
puts [pbc get -molid $molid]

# Рисуем рамку ячейки
pbc box -molid $molid -color black

# Показываем соседние периодические копии
mol showperiodic $molid 0 x
mol showperiodic $molid 0 y
mol showperiodic $molid 0 z
mol numperiodic $molid 0 1

# Получаем список фрагментов
set sel [atomselect $molid all]
set fragments [lsort -integer -unique [$sel get fragment]]

puts "Atoms: [$sel num]"
puts "Fragments: [llength $fragments]"

# Удаляем все старые representations
set nreps [molinfo $molid get numreps]
for {set r [expr {$nreps - 1}]} {$r >= 0} {incr r -1} {
    mol delrep $r $molid
}

# Каждая молекула своим цветом, шарики + связи
set i 0
foreach frag $fragments {
    mol addrep $molid
    mol modselect $i $molid "fragment $frag"

    # Шарики + связи
    mol modstyle $i $molid CPK 0.7 0.3 8 8

    # Цвет по молекуле
    mol modcolor $i $molid ColorID [expr {$i % 32}]

    incr i
}

display projection Perspective
rotate y by 110
# rotate x by 45
# rotate z by -45
scale by 1.6


$sel delete