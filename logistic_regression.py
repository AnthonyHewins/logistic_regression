from math import e
import pandas
from matplotlib import pyplot as plot
import numpy as np

#mx+b

data = pandas.read_csv('data.csv')
iterations = 100000
rate = 0.01

def sigmoid(theta, x):
	return 1.0/(1.0+(e ** -((theta[1] * x) + theta[0])))

def graph(theta):
	x = np.array(range(100))
	y = sigmoid(theta, x)
	plot.plot(x, y)
	plot.scatter(data.iloc[0:, 0], data.iloc[0:, 1], alpha=0.5, s=40)
	plot.show()

def step_gradient(theta, points):
	n = len(points)
	gradient = [0.0,0.0]
	for i in range(n):
		x = points[i,0]
		y = points[i,1]
		error = sigmoid(theta, x) - y
		gradient[0] += error
		gradient[1] += error * x
	n=float(n)
	theta[0] -= gradient[0] * rate/n
	theta[1] -= gradient[1] * rate/n
	return theta

def main():
	theta = [0.5,-1.0]
	for i in range(iterations):
		theta = step_gradient(theta,np.array(data))
	graph(theta)
	print("b =", theta[0], " m =", theta[1])

if __name__ == "__main__":
	main()
