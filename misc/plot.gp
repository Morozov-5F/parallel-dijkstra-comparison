set term pdfcairo enhanced font "CMU Serif, 60" size 33.1in, 23.4in linewidth 5

set out "plot.pdf"

set grid
set autoscale xy

set title "Various dijkstra algorithms comparison" font "CMU Serif, 80"

set xlabel  "Vertices, units"
set ylabel  "Time, seconds"

set key out

# set yrange [-0.25:3.5]

set ytics nomirror
set tics out

# Phase portrait
set size square
set style fill transparent solid 0.15 border

set key out

plot "output.dat" using 1:2 title 'Reference' w l,\
     "output.dat" using 1:3 title 'CPU (OpenMP)' w l, \
     "output.dat" using 1:4 title 'CPU (OpenCL)' w l, \
     "output.dat" using 1:5 title 'GPU (OpenCL)' w l, \
     "output.dat" using 1:6 title 'GPU (CUDA)' w l, \
     "output.dat" using 1:7 title 'GPU (OpenACC)' w l, \
