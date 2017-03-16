set title 'performance comparison'
set xlabel "N"
set ylabel "time(sec)"
set xrange [100:4100]
set terminal png
set output 'output_plot.png'
plot "result.csv" using 1:2 w lp pt 7 lc 1 title 'naive', "result.csv" using 1:3 w lp pt 7 lc 2 title 'sse', "result.csv" using 1:4 w lp pt 7 lc 3 title 'sse_prefetch'
