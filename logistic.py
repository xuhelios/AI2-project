import math
import random
from kmean import *

def logistic(z):
	h = 1 / (1 + math.exp(-z))
	if h >= 0.5:
		return h, 1
	else:
		return h, 0

def logistic_learning(training_data, training_class):
	w = [random.uniform(-1, 1), random.uniform(-1, 1)]
	alpha = 0.1  # learning rate
	b = random.uniform(-1, 1)

	for i in range(len(training_data)):
		x = training_data[i]
		y = training_class[i]
		# calculate h_w(x_i)
		z = w[0] * x[0] + w[1] * x[1] +b
		hw, predictY = logistic(z)
		# true label
		if predictY != y:
			w_0 = w[0] + alpha * (y - hw) * x[0]
			w_1 = w[1] + alpha * (y - hw) * x[1]
			b = b + alpha * (y - hw)
			w = [w_0, w_1]
	#print("After training w = ", w)
	return w, b

def test_logistic(test_data, test_class, w, b):
	misclassified = 0
	predict_label = []
	for i in range(len(test_data)):
		x = test_data[i]
		y = test_class[i]
		z = w[0] * x[0] + w[1] * x[1]+b
		h, label = logistic(z)
		if label != y:
			# print(x, " true label =  ", y, "   ", label)
			misclassified += 1
		predict_label.append(label)
	printFile("p2.dat", test_data, predict_label)
	# printGroup(test_data, test_class)
	print("Logistic Function misclassified", misclassified)
	return misclassified / len(test_data)


def logistic_algorithm(train_data, train_class, test_data, test_class):
	final_w, final_b = logistic_learning(train_data, train_class)
	error_rate = test_logistic(test_data, test_class, final_w, final_b)
	plotData(1)
	print("Logistic Function's error rate'", error_rate)
	return error_rate
