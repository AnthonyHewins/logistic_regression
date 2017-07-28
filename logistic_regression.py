import pandas
from matplotlib import pyplot as plot
import numpy as np

#mx+b

data = pandas.read_csv('data.csv')

def line(theta,x):
        return (theta[1] * x) + theta[0]

def graph(theta):
	x = np.array(range(100))	#x will be a vector of values (more concisely, 
	y = line(theta, x) 		    #y will be a vector of values
	plot.plot(x, y)
	plot.scatter(data.iloc[0:, 0], data.iloc[0:, 1], alpha=0.5, s=40)
	plot.show()

def step_gradient(theta, points, learningRate):
	gradient = [0.0,0.0]
	n = len(points)
	for i in range(n):
		x = points[i, 0]
		result = (((theta[1] * x) + theta[0]) - points[i, 1])
		gradient[0] += result
		gradient[1] += x * result
	n = float(n)
	for i in range(len(theta)):
		theta[i] -= learningRate * (gradient[i] * 2/n)
	return theta

def gradient_descent_runner(theta):
	iterations = 1000
	rate = 0.0001
	for i in range(iterations):
		b, m = step_gradient(theta, np.array(data), rate)
		if i % 100 == 0:
			graph(theta)
	return b, m

def main():
	theta  = gradient_descent_runner([0.0, 1.0])
	print("b =", theta[0], " m =", theta[1])

if __name__ == "__main__":
	main()
