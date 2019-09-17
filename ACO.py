import numpy as np
import random
import matplotlib.pyplot as plt
from operator import itemgetter

np.set_printoptions(suppress=True)

#Los costos asociados al arco ij
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

#Los τij o intensidad de feromona asociada al arco ij
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

initialPheromones = np.array([ 
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

class ant(object):

	def __init__(self):

		self.tabu = []
		self.initialNode = None
		self.currentNode = None
		self.nextNode = None
		self.pathFound = False
		self.deadEnd = False

	def searchSol(self):
		"""
        :type D: numpy array (distance matrix)
		:type P: numpy array (pheromone matrix)
		:type V: numpy array (visibility matrix)
        :rtype: string
        """
		totalCities = distances.shape[0]
		self.initialNode = random.randint(0, totalCities - 1)
		self.currentNode = self.initialNode

		while self.pathFound == False:

			if len(self.tabu) >= totalCities: 

				self.pathFound = True

			else:

				if self.deadEnd == True:
					
					self.currentNode = self.initialNode
					self.nextNode = None
					self.tabu = []
					self.deadEnd = False
					

				self.tabu.append(self.currentNode)
				self.nextNode = self.chooseNext(totalCities)

				self.currentNode = self.nextNode
				self.nextNode = None


		return (self.tabu, self.evaluateSol(totalCities))

	def chooseNext(self, nodesNum):

		candidates = []
		probabilities = []
		eta = 0
		tau = 0
		etaTau = 0
		sumEtaTau = 0

		if len(self.tabu) == nodesNum:

			self.pathFound = True

			return -1

		#Obteniendo Candidatos posibles, no visitados.
		for x in range(nodesNum):

			a, b = distances[self.currentNode][x] != 0, x not in self.tabu

			if  a == True and b == True:

				eta = visibility[self.currentNode][x]
				tau = pheromones[self.currentNode][x]
				etaTau = tau * eta
				sumEtaTau += etaTau

				toAppend = [x, eta, tau, etaTau, 0]
				candidates.append(toAppend)
			
		if len(candidates) == 0 and len(self.tabu) < nodesNum:

			self.deadEnd = True

			return -1

		#Cálculo de las probabilidades.
		for i in range (0, len(candidates)):

			candidates[i][4] = candidates[i][3] / sumEtaTau

		#Probabilidad acumulada
		candidates.sort(key = itemgetter(4), reverse = True)
		
		for i in range(0, len(candidates) - 1):

			for j in range(i + 1, len(candidates)):

					candidates[i][4] += candidates[j][4]

		
		#Selección por ruleta

		if len(candidates) == 1:

			return candidates[0][0]
		
		elif len(candidates) == 0:

			if len(self.tabu) == nodesNum:

				self.pathFound = True
				return -1

			else:

				self.deadEnd = True
				return -1

		else: 
			
			dart = random.uniform(0, 1)

			for i in range(0, len(candidates) - 1):

				if 0 < dart and dart < candidates[len(candidates) - 1][4]:

					selection = candidates[len(candidates) - 1][0]
					return selection

				elif candidates[i][4] > dart and dart > candidates[i + 1][4]:

					selection = candidates[i][0]
					return selection
	
	def evaluateSol(self, nodesNum):

		score = 0
		
		for x in range(len(self.tabu) - 1):

			i, j = self.tabu[x], self.tabu[x + 1]

			score = score + distances[i][j]

		return int(score)
		
class Solution(object):

	def trails(self, antPopulation, P):
		"""
        :type antPopulation: int
        :rtype: string
        """
		time = 150
		xaxis, yaxis, performance = [], [], []

		while time != 0: 

			colony = []
			for ants in range(antPopulation):

				colony.append(ant().searchSol())


			fittest = min(colony, key = itemgetter(1))

			xaxis.append(time)
			yaxis.append(fittest[1])


			print("time left: ", time)
			print("fittest ant: ", fittest)
			print("other scores:")

			for insect in colony:

				print(insect[1])

			self.evaporate(0.5, P)
			self.updatePheromones(fittest, P)

			time -= 1

		performance.append(xaxis)
		performance.append(yaxis)
		
		print("PHEROMONE TRAIL: \n", pheromones)

		self.plotStats(performance)

		return (fittest)
			

	def evaporate(self, rho, P):

		evaporationRate = (1.0 - rho)
		P *= evaporationRate

	def updatePheromones(self, antTrail, P):

		trail = antTrail[0]
		deltaTau = 1 / antTrail[1]

		for x in range(len(trail) - 1):

			i, j = trail[x], trail[x + 1]
			
			P[i][j] += deltaTau

	def plotStats(self, performance):

		fig = plt.figure(figsize = (10, 6))

		ax = []

		ax.append(fig.add_subplot(2, 4, 1))
		ax[-1].set_title("Distances:"+str(0))
		plt.imshow(distances, cmap= "afmhot", interpolation = "bessel")

		ax.append(fig.add_subplot(2, 4, 2))
		ax[-1].set_title("Initial Trails:")
		plt.imshow(initialPheromones, cmap= "afmhot", interpolation = "bessel")

		ax.append(fig.add_subplot(2, 4, 3))
		ax[-1].set_title("Visibility:")
		plt.imshow(visibility, cmap= "afmhot", interpolation = "bessel")

		ax.append(fig.add_subplot(2, 4, 4))
		ax[-1].set_title("Result Trails:")
		plt.imshow(pheromones, cmap= "afmhot", interpolation = "bessel")

		ax.append(fig.add_subplot(2, 1, 2))
		ax[-1].set_title("Performance over Time:")
		plt.plot(performance[0], performance[1])
		plt.show()

		#distances, initialPheromones, visibility, pheromones
		#plt.imshow(pheromones, cmap= "afmhot", interpolation = "bessel")
		plt.show()

		
		print("PERFORMANCE: ")
		print(performance)

		

def main():

	time = 0
	stepCounter = 0

	

	#print(distances)
	#print(pheromones)
	#print(visibility)

	#print(cityCounter)

	Solution().trails(4, pheromones)

if __name__ == '__main__':

	main()