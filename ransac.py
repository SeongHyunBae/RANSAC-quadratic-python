import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def data_generator():
	x_data = []
	y_data = []

	for i in range(100):
		x = np.random.randn()*5
		y = quadratic_func(x) + np.random.randint(20)
		x_data.append(x)
		y_data.append(y)

	# for noise
	for i in range(20):
		x = np.random.randn()*5
		if np.random.randn() > 0:
			y = quadratic_func(x) + np.random.randint(200)
		else:
			y = quadratic_func(x) - np.random.randint(200)

		x_data.append(x)
		y_data.append(y)

	return x_data, y_data

def quadratic_func(x):
	return 4 * (float(x)-2)**2 + 3 # y = 4x^2-16x+19


class RANSAC:
	def __init__(self, x_data, y_data, n, d):
		self.x_data = x_data
		self.y_data = y_data
		self.n = n
		self.d = d
		self.c_max = 0
		self.best_model = None

	def random_sampling(self):
		sample = []
		save_ran = []
		count = 0

		# get three points from data
		while True:
			ran = np.random.randint(len(self.x_data))

			if ran not in save_ran:
				sample.append((self.x_data[ran], self.y_data[ran]))
				save_ran.append(ran)
				count += 1

				if count == 3:
					break

		return sample

	def make_model(self, sample):
		# calculate a, b, c value from three points by using matrix
		#
		# y = ax^2 + bx + c
		# pt1 = (1, 1)
		# pt2 = (2, 4)
		# pt3 = (3, 9)
		#
		#  a +  b + c = 1
		# 4a + 2b + c = 4
		# 9a + 3b + c = 9
		#
		#  a +  b + c      1         1   1   1     a     1
		# 4a + 2b + c   =  4   =>    4   2   1  X  b  =  4
		# 9a + 3b + c      9         9   3   1     c     9
		#
		#               -1
		#  a     1  1  1      1 
		#  b  =  4  2  1   X  4   
		#  c     9  3  1      9 
		pt1 = sample[0]
		pt2 = sample[1]
		pt3 = sample[2]

		A = np.array([[pt1[0]**2, pt1[0], 1], [pt2[0]**2, pt2[0], 1], [pt3[0]**2, pt3[0], 1]]) 
		B = np.array([[pt1[1]], [pt2[1]], [pt3[1]]])
		
		inv_A = inv(A)

		return np.dot(inv_A, B)

	def eval_model(self, model):
		count = 0

		for i in range(len(self.x_data)):
			# if point is farther than d, don`t count
			if abs(self.y_data[i] - (model[0]*self.x_data[i]**2 + model[1]*self.x_data[i] + model[2])) < self.d:
				count += 1
		
		return count

	def execute_ransac(self):
		# find best model
		for i in range(self.n):
			model = self.make_model(self.random_sampling())
			
			c_temp = self.eval_model(model)
			
			if c_temp > self.c_max:
				self.best_model = model
				self.c_max = c_temp


if __name__ == '__main__':
	# make data
	x_data, y_data = data_generator()

	# show data by scatter type
	plt.scatter(x_data, y_data, c='blue', marker='o', label='data')

	# make ransac class
	# n: how many times try sampling
	# d: distance that divides inliers and outliers
	ransac = RANSAC(x_data, y_data, 50, 5)
	
	# execute ransac algorithm
	ransac.execute_ransac()
	
	# get best model from ransac
	a, b, c = ransac.best_model[0][0], ransac.best_model[1][0], ransac.best_model[2][0]

	# make function with result
	f = lambda x : a*x**2 + b*x + c

	x_result = np.arange(np.min(ransac.x_data), np.max(ransac.x_data)+1)
	y_result = []

	for i in x_result:
		y_result.append(f(i))
	
	# show result by plot type
	plt.plot(x_result, y_result, color='red')

	plt.tight_layout()
	plt.show()
