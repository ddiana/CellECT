# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
from numpy import random
from scipy import io
import pdb
#from PyML import VectorDataSet
#from PyML.classifiers.svm import SVM, loadSVM
#from PyML.classifiers.svm import loadSVM
from sklearn import svm
import os
from termcolor import colored
import sys
import copy

# Imports from this project
from CellECT.seg_tool.seg_io import load_all
from CellECT.seg_tool.nuclei_collection import nuclei_collection as nc
from CellECT.seg_tool.features import segment_features as feat
from CellECT.seg_tool.seg_utils import call_silent

import CellECT.seg_tool.globals




"""
Extract features from segments in the training set.
Learn SVM classifier with these features.
Classify segments from the test set according to celness confidence.
"""


class CellnessMetric(object):

	def __init__(self):

		self.generic_features = CellECT.seg_tool.globals.generic_features
		self.specific_features = CellECT.seg_tool.globals.specific_features

		self.positive_examples_generic = []
		self.negative_examples_generic = []
		self.positive_examples_specific = []
		self.negative_examples_specific = []

		self.mean_per_dim_specific = []
		self.std_per_dim_specific = []
		self.mean_per_dim_generic = []
		self.std_per_dim_generic = []


		self.generic_classifier = None
		self.specific_classifier = None

	def add_training_examples(self,list_of_segments, list_of_labels, generic_only = True):

				
		for segment, label in zip(list_of_segments, list_of_labels):

			if label == "Correct":
				self.positive_examples_generic.append(self.extract_features_from_segment(segment, "generic"))
			else:
				self.negative_examples_generic.append(self.extract_features_from_segment(segment, "generic"))


			if not generic_only:
				if label == "Correct":
					self.positive_examples_specific.append(self.extract_features_from_segment(segment, "specific"))
				else:
					self.negative_examples_specific.append(self.extract_features_from_segment(segment, "specific"))


	def extract_features_from_segment(self, segment, feature_type="generic"):

		assert (feature_type in ["generic", "specific"])

		vector = []
		features_of_interest = None

		if feature_type == "generic":
			features_of_interest = self.generic_features
		if feature_type == "specific":
			features_of_interest = self.specific_features
	
		for feature_name in features_of_interest:
			if segment.feature_dict.has_key(feature_name):
				if isinstance(segment.feature_dict[feature_name], list):
					vector.extend (segment.feature_dict[feature_name])
				else:
					vector.append (segment.feature_dict[feature_name])
		return vector

	def train(self):

		

		self.train_generic()
		self.train_specific()



	def normalize_train_data(self, data_vector, clf_type = "generic"):
		"""
		Normalize the train vectors with features for SVM.
		"""
		assert(clf_type in ["generic", "specific"])

		if clf_type == "generic":
			self.mean_per_dim_generic = []
			mean_per_dim = self.mean_per_dim_generic
			self.std_per_dim_generic = []
			std_per_dim = self.std_per_dim_generic
		else:
			self.mean_per_dim_specific = []
			mean_per_dim = self.mean_per_dim_specific
			self.std_per_dim_specific = []
			std_per_dim = self.std_per_dim_specific

		per_dim = zip(*data_vector)

		for i in xrange(len(per_dim)):
		
			m = np.float64(sum (per_dim[i]) / float (len(per_dim[i])))
			s = np.std(per_dim[i])
			per_dim[i] -= m
			if s>0:
				per_dim[i] /= s
		
			mean_per_dim.append(m)
			std_per_dim.append(s)
	
		data_vector = zip(*per_dim)
		for i in xrange(len(data_vector)):
			data_vector[i] = list(data_vector[i])

		return data_vector
		
		


	def train_generic(self):

		if len(self.positive_examples_generic) ==0 or len(self.negative_examples_generic)==0:
			return

		data = copy.deepcopy(self.positive_examples_generic)
		data.extend(self.negative_examples_generic)
		data = self.normalize_train_data(data, "generic")

		labels = ["Correct" for i in xrange(len(self.positive_examples_generic))]
		labels.extend ( ["Incorrect" for i in xrange(len(self.negative_examples_generic))])

		if len(data):
			self.generic_classifier = svm.SVC()
			self.generic_classifier.fit(data,labels)

	def train_specific(self):
	
		if len(self.positive_examples_specific) == 0 or  len(self.negative_examples_specific)==0:
			return

		data = copy.deepcopy(self.positive_examples_specific)
		data.extend(self.negative_examples_specific)
		data = self.normalize_train_data(data, "specific")

		labels = ["Correct" for i in xrange(len(self.positive_examples_specific))]
		labels.extend( ["Incorrect" for i in xrange(len(self.negative_examples_specific))])

		if len(data):
			self.specific_classifier = svm.SVC()
			self.specific_classifier.fit(data,labels)

	def get_prediction(self, data, clf_type = "generic"):

		assert (clf_type in ["generic", "specific"])

		classifier = None
		if clf_type == "specific":
			if len(self.positive_examples_specific) > 0 and  len(self.negative_examples_specific)>0:
				classifier = self.specific_classifie
		else:
			if len(self.positive_examples_generic) > 0 and  len(self.negative_examples_generic)>0:
				classifier = self.generic_classifier

		prediction = None
		discriminant_value = None

		if classifier:
			prediction = classifier.predict(data)[0]
			discriminant_value = classifier.decision_function(data)[0][0]

		return prediction, discriminant_value


	def normalize_test_vector(self, data_vector, clf_type = "generic"):

		"""
		Normalize the test vectors with features for SVM.
		"""

		assert(clf_type in ["generic", "specific"])

		if clf_type == "generic":
			mean_per_dim = self.mean_per_dim_generic
			std_per_dim = self.std_per_dim_generic
		else:
			mean_per_dim = self.mean_per_dim_specific
			std_per_dim = self.std_per_dim_specific


		for i in xrange(len(mean_per_dim)):
			data_vector[i] -= mean_per_dim[i]
			data_vector[i] /= std_per_dim[i]
		
	
		return data_vector

	
		
		
	
	def test_segment(self, segment):

		generic_feat = self.extract_features_from_segment(segment,"generic" )
		specific_feat = self.extract_features_from_segment(segment, "specific")	
		generic_feat = self.normalize_test_vector(generic_feat, "generic")
		specific_feat = self.normalize_test_vector(specific_feat, "specific")

		pred_generic, disc_generic = self.get_prediction(generic_feat, "generic")
		pred_specific, disc_specific = self.get_prediction(specific_feat, "specific")

		if pred_generic is None and pred_specific is None:
			return "Correct", 0
	
		if pred_generic is not None and pred_specific is None:
			return pred_generic, disc_generic
	
		if pred_generic is  None and pred_specific is not None:
			return pred_specific, disc_specific

		# TODO: merge classifier info

		examples_generic = len(self.positive_examples_generic) + len(self.negative_examples_generic)
		examples_specific = len(self.positive_examples_specific) + len(self.negative_examples_specific)

		p_generic = examples_generic/float(examples_specific + examples_generic)
		p_specific = examples_specific/float(examples_specific + examples_generic)

		disc = disc_generic * p_generic + disc_generic * p_specific
		if disc >0:
			return "Correct", disc
		else:
			return "Incorrect", disc


def load_generic_training_data(cellness_metric):


	# Reading trainign volume
	try:
		vol = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_var"])
	except Exception as err:
		print colored("Error: %s" % err.message, "red")
		sys.exit()
		

	# Reading training nuclei
	nuclei_collection = nc.NucleusCollection(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_var"])
	

	#### positive segment collection

	try:	
		ground_truth = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_var"])
		list_of_labels_in_gt = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_var"])
	except Exception as err:
		print colored("Error: %s" % err.message, "red")
		sys.exit()

	# converting to right format
	
	list_of_labels_in_gt = [elem[0] for elem in list_of_labels_in_gt]
	

	np.random.seed(1234)
	np.random.shuffle(list_of_labels_in_gt)

	set_of_labels_for_positive_examples = set(list_of_labels_in_gt[:30])
	collection_of_positive_segments = feat.get_segments_with_features(vol, ground_truth, set_of_labels_for_positive_examples, "ground_truth", nuclei_collection)

	

#	class_prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#	list_of_labels = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287]


	del ground_truth
	
	#### negative segment collection

	try:
		bad_watershed = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_var"])
		set_of_labels_for_negative_examples = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_var"])
	except Exception as err:
		print colored("Error: %s" % err.message, "red")
		sys.exit()

	set_of_labels_for_negative_examples = [elem[0] for elem in set_of_labels_for_negative_examples]
	set_of_labels_for_negative_examples = set(np.unique(set_of_labels_for_negative_examples))
	

	collection_of_negative_segments = feat.get_segments_with_features(vol, bad_watershed, set_of_labels_for_negative_examples, "bad_watershed", nuclei_collection)

	del bad_watershed

	#save_training_data_in_mat(collection_of_positive_segments, collection_of_negative_segments)



	cellness_metric.add_training_examples(collection_of_negative_segments.list_of_segments, ["Incorrect" for i in xrange(len(collection_of_negative_segments.list_of_segments))], True)
	cellness_metric.add_training_examples(collection_of_positive_segments.list_of_segments, ["Correct" for i in xrange(len(collection_of_positive_segments.list_of_segments))], True)







	


#def learn_classifier():

#	"""
#	Extract features from segments in the training set.
#	Learn SVM classifier with these features.
#	Return trained classifier.
#	"""

#	# Reading trainign volume
#	try:
#		vol = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_var"])
#	except Exception as err:
#		print colored("Error: %s" % err.message, "red")
#		sys.exit()
#		

#	# Reading training nuclei
#	nuclei_collection = nc.NucleusCollection(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_var"])
#	

#	#### positive segment collection

#	try:	
#		ground_truth = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_var"])
#		list_of_labels_in_gt = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_var"])
#	except Exception as err:
#		print colored("Error: %s" % err.message, "red")
#		sys.exit()

#	# converting to right format
#	
#	list_of_labels_in_gt = [elem[0] for elem in list_of_labels_in_gt]
#	

#	np.random.seed(1234)
#	np.random.shuffle(list_of_labels_in_gt)

#	set_of_labels_for_positive_examples = set(list_of_labels_in_gt[:30])
#	collection_of_positive_segments = feat.get_segments_with_features(vol, ground_truth, set_of_labels_for_positive_examples, "ground_truth", nuclei_collection)

#	

##	class_prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
##	list_of_labels = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287]


#	del ground_truth
#	
#	#### negative segment collection

#	try:
#		bad_watershed = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_var"])
#		set_of_labels_for_negative_examples = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_var"])
#	except Exception as err:
#		print colored("Error: %s" % err.message, "red")
#		sys.exit()

#	set_of_labels_for_negative_examples = [elem[0] for elem in set_of_labels_for_negative_examples]
#	set_of_labels_for_negative_examples = set(np.unique(set_of_labels_for_negative_examples))
#	

#	collection_of_negative_segments = feat.get_segments_with_features(vol, bad_watershed, set_of_labels_for_negative_examples, "bad_watershed", nuclei_collection)

#	del bad_watershed

#	#save_training_data_in_mat(collection_of_positive_segments, collection_of_negative_segments)

#	data, labels = prepare_training_data(collection_of_positive_segments, collection_of_negative_segments)


#	print "Training classifier..."


#	classifier = call_silent.call_silent_process(prepare_SVM, data, labels)

#	return classifier, data, labels

	
#def classify_segments(classifier, data_vector):

#	"""
#	Classify segments from the test set according to celness confidence.
#	Return the class prediction and SVM discriminant value.
#	"""	


#	class_prediction = []
#	discriminant_value = []
#	
#	#call_silent.call_silent_process(data_vector.attachKernel,"gaussian")
#	
#	print "Classifying new data...               "
#	for i in xrange (len(data_vector)):
#		#prediction = call_silent.call_silent_process(classifier.classify,data_vector,i)
#		prediction = classifier.predict(data_vector[i])[0]
#		class_prediction.append(prediction)
#		discriminant_value.append(classifier.decision_function(data_vector[i])[0][0])   #prediction[1])
#		#if class_prediction[-1] == 1:
#		#	print class_prediction[i]	,": ", discriminant_value[-1]	
#	
#	#number_bad_segments = np.sum(class_prediction>0)
#	#print "... Number bad segments:", number_bad_segments
#		
#	return (class_prediction, discriminant_value)



#def get_segment_feature_vector(segment, generic_only = True):
#	vector = []

#	pdb.set_trace()
#	if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
#		vector.extend (segment.feature_dict["border_to_nucleus_distance_hist"])
#		vector.append (segment.feature_dict["border_to_nucleus_distance_mean"])
#		vector.append (segment.feature_dict["border_to_nucleus_distance_std"])
#	if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_size"]) and not generic_only:
#		vector.append (segment.feature_dict["size"])
#	if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
#		vector.append (segment.feature_dict["border_to_interior_intensity_ratio"])

#	return vector


#def prepare_test_data(collection_of_segments):


#	"""
#	Extract features for each segment in the collection of segments in the test dataset.
#	"""

#	print "Preparing test data...           "

#	test_vectors = []

#	for segment in collection_of_segments.list_of_segments:
#		vector = get_segment_feature_vector(segment)
#		test_vectors.append(vector)



#	if not CellECT.seg_tool.globals.DEFAULT_PARAMETER["no_cellness_metric"]:
#		test_vectors = normalize_test_data(test_vectors)	
#		
#	
#	return test_vectors


#def normalize_test_data( test_vectors):

#	"""
#	Normalize the test vectors with features for SVM.
#	"""

#	per_dim =  zip(*test_vectors)

#	for i in xrange(len(per_dim)):
#		per_dim[i] -= CellECT.seg_tool.globals._mean_per_dim[i]
#		per_dim[i] /= CellECT.seg_tool.globals._std_per_dim[i]
#		

#	test_vectors = zip(*per_dim)

#	for i in xrange(len(test_vectors)):
#		test_vectors[i] = list(test_vectors[i])
#	
#	
#	return test_vectors
#		



def prepare_training_data(collection_of_positive_segments, collection_of_negative_segments):

	"""
	Extract features for each segment in the collection of segments in the train dataset.
	"""


	print "Preparing training data..."

	training_vectors = []
	training_labels = []


	for segment in collection_of_positive_segments.list_of_segments:
		vector = get_segment_feature_vector(segment)

		training_labels.append("Correct")
		training_vectors.append(vector)
		
		
	for segment in collection_of_negative_segments.list_of_segments:
		vector = get_segment_feature_vector(segment)
		training_labels.append("Incorrect")
		training_vectors.append(vector)
		
	
	training_vectors = normalize_train_data(training_vectors)
	
		
	#data = VectorDataSet(training_vectors,L=training_labels)
		

#	return training_vectors, training_labels
#	
#def prepare_SVM(data,labels):


#	
#	classifier = svm.SVC(kernel="rbf")
#	classifier.fit(data, labels)


##	data.attachKernel("gaussian")
##	classif = SVM(optimizer="libsvm")
##	
##	classif.train(data)


#	return classifier



#def save_training_data_in_mat(col_pos, col_neg):

#	if int(globals.DEFAULT_PARAMETER["use_border_intensity"]):
#		border_intensity_pos = col_pos.get_feature_values_in_list("border_to_interior_intensity_ratio")
#		border_intensity_neg = col_neg.get_feature_values_in_list("border_to_interior_intensity_ratio")

#	if int(globals.DEFAULT_PARAMETER["use_border_distance"]):
#		border_dist_mean_pos = col_pos.get_feature_values_in_list("border_to_nucleus_distance_mean")
#		border_dist_mean_neg = col_neg.get_feature_values_in_list("border_to_nucleus_distance_mean")

#		border_dist_std_pos = col_pos.get_feature_values_in_list("border_to_nucleus_distance_std")
#		border_dist_std_neg = col_neg.get_feature_values_in_list("border_to_nucleus_distance_std")

#		border_dist_hist_pos = col_pos.get_feature_values_in_list("border_to_nucleus_distance_hist")
#		border_dist_hist_neg = col_neg.get_feature_values_in_list("border_to_nucleus_distance_hist")

#	if int(globals.DEFAULT_PARAMETER["use_size"]):
#		size_pos = col_pos.get_feature_values_in_list("size")
#		size_neg = col_neg.get_feature_values_in_list("size")


##	plot_features(border_intensity_pos, size_pos, border_intensity_neg, size_neg, "Border To Interior Intensity", "Size")
##	plot_features(border_dist_mean_pos, border_dist_std_pos, border_dist_mean_neg, border_dist_std_neg, "Border Distance To Nucleus, Mean", "Border Distance To Nucleus, StDev")

#	pdb.set_trace()

#	io.savemat("training.mat", {"border_intensity_pos": border_intensity_pos,"border_intensity_neg":border_intensity_neg,"border_dist_mean_pos":border_dist_mean_pos,"border_dist_mean_neg":border_dist_mean_neg,"border_dist_std_pos":border_dist_std_pos,"border_dist_std_neg":border_dist_std_neg,"size_pos":size_pos,"size_neg":size_neg})



#def plot_features(vec1pos, vec2pos, vec1neg, vec2neg, name1, name2):

#	pylab.plot(vec1pos, vec2pos, "*b", label = "Correct Segments")
#	pylab.hold(True)
#	pylab.plot(vec1neg, vec2neg, "or", label ="Incorrect Segments")
#	pylab.xlabel(name1)
#	pylab.ylabel(name2)
#	pylab.legend(numpoints=1)
#	pylab.show()



