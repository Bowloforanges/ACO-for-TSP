import numpy as np
import random
from operator import itemgetter

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


		return self.tabu



	def chooseNext(self,nodesNum):

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

				
			

		
		


			
		

def main():

	time = 0
	stepCounter = 0

	

	#print(distances)
	#print(pheromones)
	#print(visibility)

	#print(cityCounter)

	print(ant().searchSol())




if __name__ == '__main__':

	main()