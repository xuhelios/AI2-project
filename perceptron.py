import numpy as np
import math
import random
from kmean import *
#
# def plotData():
# 	# g = Gnuplot.Gnuplot(persist =1)
#
# 	gp.c("set palette model RGB defined ( 0 'red', 1 'green', 2 'blue', 3 'black')")
# 	gp.c('plot "data1.dat" using 2:3:1 with points palette')


def threshold(w, x, b):

	t = w[0] * x[0] + w[1] * x[1] + b
	if t >= 0:
		return 1
	else:
		return 0


def perceptron_learning(data, class_label):
	w = [random.uniform(-1, 1), random.uniform(-1, 1)]
	alpha = 0.1  # learning rate
	b = random.uniform(-1, 1)
	for i in range(len(data)):
		x = data[i]
		y = class_label[i]
		h = threshold(w, x, b)
		if h != y:
			w_0 = w[0] + alpha * (y - h) * x[0]
			w_1 = w[1] + alpha * (y - h) * x[1]
			b = b + alpha * (y - h)
			w = [w_0, w_1]

	return w, b



def test_perceptron(data, class_label, w, b):
	misclassified = 0
	out_label = []
	for i in range(len(data)):
		x = data[i]
		y = class_label[i]
		label = threshold(w, x, b)
		if label != y:
			misclassified += 1
		out_label.append(label)
	print("Perceptron misclassified", misclassified)
	# printGroup(data, class_label)

	# VISUALIZATION
	printFile("p1.dat", data, out_label)
	f = "f(x) = ( -x*"+str(w[0]) + "-" +str(b) + ")/"+ str(w[1])
	gp.c('set terminal qt 1 size 500,400 title "Perceptron"')
	gp.c('set xrange [-2:2]')
	gp.c('set yrange [-2:2]')
	gp.c("set palette model RGB defined ( 0 'red', 1 'green', 2 'blue', 3 'black')")
	gp.c("unset key")
	file1 = "p1.dat"
	gp.c(f)
	gp.c('plot "' + file1 + '" using 2:3:1 with points palette, f(x)')
	return misclassified/len(data)


def perceptron(train_data, train_class, test_data, test_class):
	final_w, final_b = perceptron_learning(train_data, train_class)
	error_rate = test_perceptron(test_data, test_class, final_w, final_b)
	print("Perceptron's error rate'", error_rate)
	return error_rate

