set title "Simple Plots" font ",20"
set terminal wxt
unset key
set style data points
set palette model RGB defined ( 0 'red', 1 'green', 2 'blue', 3 'black' )
plot "data1.dat" using 2:3:1 with points palette
set terminal wxt 1 

unset key
set style data points
set palette model RGB defined ( 0 'red', 1 'green', 2 'blue', 3 'black' )
set xrange [-2:2]
set yrange [-2:2]
plot "data1.dat" using 2:3:1 with points palette, -4*x-1 
