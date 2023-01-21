set style data histograms
set style fill solid
set boxwidth 1
set terminal png size 1000,600

set xlabel "Flags d'optimisation" font ",15"
set ylabel "GFLOP/s" font ",15"

set title "Flags d'optimisation pour la version de base" font ",15"
set output "flag.png"
plot "flag.dat" using 2:xtic(1) title "GCC" lt rgb "blue",\
     "" using 3 title "CLANG" lt rgb "green",

##############################################################

set xlabel "Les Versions" font ",15"
set ylabel "GFLOP/s" font ",15"
set yrange [0:130]

set title "Comparaison entre les versions" font ",15"
set output "version.png"

plot "version.dat" using 2:xtic(1) title "GCC" lt rgb "blue",\
     "" using 3 title "CLANG" lt rgb "green",


##############################################################

set xlabel "Les Caches(Memory size)" font ",15"
set ylabel "GFLOP/s" font ",15"
set yrange [0:130]

set title "Cache Levels" font ",15"
set output "cache.png"

plot "cache.dat" using 2:xtic(1) title "GCC" lt rgb "blue",\
     "" using 3 title "CLANG" lt rgb "green",

