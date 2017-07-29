from math import e
import pandas
from matplotlib import pyplot as plot
import numpy as np

#mx+b

data = pandas.read_csv('data.csv')
iterations = 100
rate = 0.0001
theta = [0.0,0.0]

def sigmoid(x):
	return (1/(1+(e ** -(line(x)))))

def line(x):
	return (theta[1] * x) + theta[0]

def graph():
	x = np.array(range(100))	#x will be a vector of values
	y = sigmoid(x)		    	#y will be a vector of values
	plot.plot(x, y)
	plot.scatter(data.iloc[0:, 0], data.iloc[0:, 1], alpha=0.5, s=40)
	plot.show()

def step_gradient(theta, points):
	n = len(points)
	gradient = [0.0,0.0]
	for i in range(n):
		x = points[i,0]
		y = points[i,1]
		error = sigmoid(x) - y
		gradient[0] += error
		gradient[1] += error * x
	n=float(n)
	theta[0] -= gradient[0] * rate/n
	theta[1] -= gradient[1] * rate/n

def main():
	theta = [3, 4]
	graph()
	'''for i in range(iterations):
		step_gradient(theta,np.array(data))
		if i % 100: graph()
	print("b =", theta[0], " m =", theta[1])'''

if __name__ == "__main__":
	main()
