from logistic import *
from perceptron import *
import numpy as np
import random


def back_prop_learning(ex_data, ex_label):
	n = random.randint(5, 10)
	w = [[0 for j in range(n+3)] for i in range(n+2)]
	b = [0 for j in range(n + 3)]
	layer =[[0,1],[i for i in range(2, n+2)], [n+2]]
	alpha = 0.1
	#2 nodes in input layer, those are coordinates of points
	#n nodes in hidden layer
	for i in range(n):
		for j in range(n+3):
			w[i][j] = random.uniform(-1, 1)
	for j in range(n + 3):
		b[j] = random.uniform(-1, 1)

	for k in range(len(ex_data)):
		x = ex_data[k]
		y = ex_label[k]

		a = [0 for i in range(2+n+1)]
		delta = [0 for i in range(2 + n + 1)]
		vector_in = [0 for i in range(2 + n + 1)]
		# print("x = ", x )
		for i in range(2):
			a[i] = x[i]

		# for each node j in hidden layer
		# n nodes in the hidden layer are numbered from 0 -> n + 1#for each node j in hidden layer
		#         # n nodes in the hidden layer are numbered from 0 -> n + 1
		for j in range(2, n+2):
			vector_in[j] = 0
			for i in range(2):
				vector_in[j] = vector_in[j] + w[i][j]*a[i]
			a[j] = logistic(vector_in[j] + b[j])[1]

		#calculate output node
		j = n+2
		vector_in[j] = 0.0
		for i in range(2, n + 2):
			vector_in[j] = vector_in[j] + w[i][j] * a[i]
		a[n+2] = logistic(vector_in[j] + b[n+2])[1]

		delta[j] = y - a[n+2]

		for l in range(1,-1,-1):
			for i in layer[l]:
				g = logistic(vector_in[i] + b[i])[1]
				if l == 1:
					delta[i] = g*(1-g)*w[i][j]*delta[n+2] #one output node j
				if l == 0:
					delta[i] = g*(1-g)*sum([w[i][j]*delta[j] for j in range(2, n+2)]) #one output node j

		for j in range(2, n+2): #foreach j in 2-> n+1
			#from first input layer to hidden layer
			w[0][j] = w[0][j] + alpha*a[0]*delta[j]
			w[1][j] = w[1][j] + alpha * a[1] * delta[j]

			#from hidden layer to output layer
			w[j][n+2] = w[j][n+2] + alpha * a[j] * delta[n+2]

		for j in range(2, n+3):
			b[j] = b[j] + alpha*delta[j]

	return w, b, n


def test_network(test_data, test_label, w, b, n):

	misclassified = 0
	predict_label = []

	for k in range(len(test_data)):
		x = test_data[k]
		y = test_label[k]
		# 2 nodes of input layer, n nodes in hidden layer, 1 node in output layer
		a = [0 for i in range(n+3)]

		for i in range(2):
			a[i] = x[i]

		#calculate nodes in hidden layer
		for j in range(2, n+2):
			a[j] = logistic(a[0]*w[0][j] + a[1]*w[1][j] + b[j])[1]

		#calcalate output node
		sumout = sum([w[j][n+2]*a[j] for j in range(2, n+2)]) + b[n+2]
		a[n+2] = logistic(sumout)[1]

		if a[n+2] != y:
			misclassified +=1
		predict_label.append(a[n+2])
	print("Neural Network misclassified", misclassified)
	printFile("p2.dat", test_data, predict_label)
	plotData(2)


	return misclassified/len(test_data)


def single_layer_neural_network(new_training_data, new_training_class, new_test_data, new_test_class):
	w, b, n = back_prop_learning(new_training_data, new_training_class)
	error_rate_network = test_network(new_test_data, new_test_class, w, b, n)
	print("Neural Network's error rate ", error_rate_network)
	return error_rate_network







