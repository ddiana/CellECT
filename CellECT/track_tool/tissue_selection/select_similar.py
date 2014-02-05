# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from sklearn import svm
from sklearn import preprocessing
import pdb
import numpy as np


# Imports from this project



class TissueSelector(object):


	def __init__(self):

		self.classifier = None
		self.training_vectors = []
		self.mean_per_dim = []
		self.std_per_dim = []

	def add_training_examples(self, data_vector):

		self.training_vectors.extend(data_vector)


	def test_sample(self, test_vector):

		self.normalize_test_sample(test_vector)


		return self.classifier.decision_function(test_vector)[0][0]



	
	def train(self):


	
		normed_data = self.normalize_data_vector()

		self.classifier = svm.OneClassSVM()

		self.classifier.fit(normed_data)


	def normalize_test_sample(self, data_vector):


		for i in xrange(len(self.mean_per_dim)):

			data_vector[i] -= self.mean_per_dim[i]
			if self.std_per_dim[i]>0:
				data_vector[i] /= self.std_per_dim[i]




		preprocessing.normalize([data_vector], norm='l2')

	def normalize_data_vector(self):

		if len(self.training_vectors) == 0:
			return			


		data_vector = self.training_vectors
		per_dim = zip(*data_vector)

		for i in xrange(len(per_dim)):
		
			m = np.float64(sum (per_dim[i]) / float (len(per_dim[i])))
			s = np.std(per_dim[i])

			if s>0:
				per_dim[i] /= s
		
			self.mean_per_dim.append(m)
			self.std_per_dim.append(s)
	
		data_vector = zip(*per_dim)
		for i in xrange(len(data_vector)):
			data_vector[i] = list(data_vector[i])

		preprocessing.normalize(data_vector, norm='l2')
		return data_vector



