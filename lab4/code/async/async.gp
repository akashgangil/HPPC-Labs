set term png
set output 'async.png'

set title "MPI_Isend Overlap Latency"
set xlabel "Work Delay (seconds)"
set ylabel "Time (s)"
set grid
#set logscale xy
plot "async.dat" notitle with linespoints

# eof
