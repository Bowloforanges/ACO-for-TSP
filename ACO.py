import numpy as np


class ant(object):
	def __init__(self):

		tabu = []
		currentNode = None
		nextNode = None

	def searchSol(self, D, P, V):
		"""
        :type D: numpy array (distance matrix)
		:type P: numpy array (pheromone matrix)
		:type V: numpy array (visibility matrix)
        :rtype: string
        """
		return "\nFUCK OFF MATE, NOT READY YET!"





def main():

	distances = np.array([
		(0, 8, 7, 0, 3, 0, 0, 0, 0, 7, 5),
		(8, 0, 0, 0, 0, 4, 0, 0, 4, 5, 4), 
		(7, 0, 0, 0, 6, 0, 0, 7, 0, 0, 0),
		(0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0),
		(3, 0, 6, 0, 0, 5, 0, 5, 0, 0, 4),
		(0, 4, 0, 0, 5, 0, 3, 5, 3, 0, 1),
		(0, 0, 0, 2, 0, 3, 0, 5, 3, 0, 0),
		(0, 0, 7, 0, 5, 5, 5, 0, 0, 0, 0),
		(0, 4, 0, 3, 0, 3, 3, 0, 0, 0, 0),
		(7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		(5, 4, 0, 0, 4, 1, 0, 0, 0, 0, 0)
		])

	pheromones = np.array([
		(0, 0.01, 0.01, 0, 0.01, 0, 0, 0, 0, 0.01, 0.01),
		(0.01, 0, 0, 0, 0, 0.01, 0, 0, 0.01, 0.01, 0.01), 
		(0.01, 0, 0, 0, 0.01, 0, 0, 0.01, 0, 0, 0),
		(0, 0, 0, 0, 0, 0, 0.01, 0, 0.01, 0, 0),
		(0.01, 0, 0.01, 0, 0, 0.01, 0, 0.01, 0, 0, 0.01),
		(0, 0.01, 0, 0, 0.01, 0, 0.01, 0.01, 0.01, 0, 0.01),
		(0, 0, 0, 0.01, 0, 0.01, 0, 0.01, 0.01, 0, 0),
		(0, 0, 0.01, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0),
		(0, 0.01, 0, 0.01, 0, 0.01, 0.01, 0, 0, 0, 0),
		(0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		(0.01, 0.01, 0, 0, 0.01, 0.01, 0, 0, 0, 0, 0)
		])

	visibility = np.array([
		(0, 1/8, 1/7, 0, 1/3, 0, 0, 0, 0, 1/7, 1/5),
		(1/8, 0, 0, 0, 0, 1/4, 0, 0, 1/4, 1/5, 1/4), 
		(1/7, 0, 0, 0, 1/6, 0, 0, 1/7, 0, 0, 0),
		(0, 0, 0, 0, 0, 0, 1/2, 0, 1/3, 0, 0),
		(1/3, 0, 1/6, 0, 0, 1/5, 0, 1/5, 0, 0, 1/4),
		(0, 1/4, 0, 0, 1/5, 0, 1/3, 1/5, 1/3, 0, 1),
		(0, 0, 0, 1/2, 0, 1/3, 0, 1/5, 1/3, 0, 0),
		(0, 0, 1/7, 0, 1/5, 1/5, 1/5, 0, 0, 0, 0),
		(0, 1/4, 0, 1/3, 0, 1/3, 1/3, 0, 0, 0, 0),
		(1/7, 1/5, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		(1/5, 1/4, 0, 0, 1/4, 1/1, 0, 0, 0, 0, 0)
		])

	time = 0
	stepCounter = 0

	cityCounter = distances.shape[0]

	print(distances)
	print(pheromones)
	print(visibility)

	print(cityCounter)

	print(ant().searchSol(distances, pheromones, visibility))


if __name__ == '__main__':

	main()