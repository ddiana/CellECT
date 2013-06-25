# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
from numpy import random
from scipy import io
import pdb
import pylab
from PyML import VectorDataSet
from PyML.classifiers.svm import SVM, loadSVM
from matplotlib.widgets import Slider
from PyML.classifiers.svm import loadSVM
import os

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


def learn_classifier():

	"""
	Extract features from segments in the training set.
	Learn SVM classifier with these features.
	Return trained classifier.
	"""


	# Reading trainign volume
	vol = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_var"])
	
	# Reading training nuclei
	nuclei_collection = nc.NucleusCollection(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_var"])
	

	#### positive segment collection
	
	ground_truth = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_var"])
	

	list_of_labels_in_gt = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_var"])
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

	bad_watershed = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_var"])
	set_of_labels_for_negative_examples = load_all.load_from_mat(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_path"], CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_var"])
	set_of_labels_for_negative_examples = [elem[0] for elem in set_of_labels_for_negative_examples]
	set_of_labels_for_negative_examples = set(np.unique(set_of_labels_for_negative_examples))
	

	collection_of_negative_segments = feat.get_segments_with_features(vol, bad_watershed, set_of_labels_for_negative_examples, "bad_watershed", nuclei_collection)

	del bad_watershed

	#save_training_data_in_mat(collection_of_positive_segments, collection_of_negative_segments)

	data = prepare_training_data(collection_of_positive_segments, collection_of_negative_segments)


	print "Training classifier..."

	classifier = call_silent.call_silent_process(prepare_SVM, data)

	return classifier

	
def classify_segments(classifier, data_vector):

	"""
	Classify segments from the test set according to celness confidence.
	Return the class prediction and SVM discriminant value.
	"""	

	class_prediction = []
	discriminant_value = []
	
	call_silent.call_silent_process(data_vector.attachKernel,"gaussian")
	
	print "Classifying new data...               "
	for i in xrange (len(data_vector)):
		prediction = call_silent.call_silent_process(classifier.classify,data_vector,i)
		class_prediction.append(prediction[0])
		discriminant_value.append(prediction[1])
		#if class_prediction[-1] == 1:
		#	print class_prediction[i]	,": ", discriminant_value[-1]	
	
	#number_bad_segments = np.sum(class_prediction>0)
	#print "... Number bad segments:", number_bad_segments
		
	return (class_prediction, discriminant_value)



def prepare_test_data(collection_of_segments):


	"""
	Extract features for each segment in the collection of segments in the test dataset.
	"""

	print "Preparing test data...           "

	test_vectors = []

	for segment in collection_of_segments.list_of_segments:
		vector = []
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
			vector.extend (segment.feature_dict["border_to_nucleus_distance_hist"])
			vector.append (segment.feature_dict["border_to_nucleus_distance_mean"])
			vector.append (segment.feature_dict["border_to_nucleus_distance_std"])
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_size"]):
			vector.append (segment.feature_dict["size"])
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
			vector.append (segment.feature_dict["border_to_interior_intensity_ratio"])
		test_vectors.append(vector)
		
	test_vectors = normalize_test_data(test_vectors)	
		
	
	return test_vectors


def normalize_test_data( test_vectors):

	"""
	Normalize the test vectors with features for SVM.
	"""

	per_dim =  zip(*test_vectors)

	for i in xrange(len(per_dim)):
		per_dim[i] -= CellECT.seg_tool.globals._mean_per_dim[i]
		per_dim[i] /= CellECT.seg_tool.globals._std_per_dim[i]
		

	test_vectors = zip(*per_dim)

	for i in xrange(len(test_vectors)):
		test_vectors[i] = list(test_vectors[i])
	
	
	return test_vectors
		


def normalize_train_data( training_vectors):

	"""
	Normalize the train vectors with features for SVM.
	"""

	per_dim = zip(*training_vectors)

	for i in xrange(len(per_dim)):
		
		m = np.float64(sum (per_dim[i]) / float (len(per_dim[i])))
		s = np.std(per_dim[i])
		per_dim[i] -= m
		per_dim[i] /= s
		
		CellECT.seg_tool.globals._mean_per_dim.append(m)
		CellECT.seg_tool.globals._std_per_dim.append(s)
	
	training_vectors = zip(*per_dim)
	for i in xrange(len(training_vectors)):
		training_vectors[i] = list(training_vectors[i])

	return training_vectors
		

def prepare_training_data(collection_of_positive_segments, collection_of_negative_segments):

	"""
	Extract features for each segment in the collection of segments in the train dataset.
	"""


	print "Preparing training data..."

	training_vectors = []
	training_labels = []


	for segment in collection_of_positive_segments.list_of_segments:
		vector = []
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
			vector.extend (segment.feature_dict["border_to_nucleus_distance_hist"])
			vector.append (segment.feature_dict["border_to_nucleus_distance_mean"])
			vector.append (segment.feature_dict["border_to_nucleus_distance_std"])
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_size"]):
			vector.append (segment.feature_dict["size"])
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_intensity"]):
			vector.append (segment.feature_dict["border_to_interior_intensity_ratio"])

		training_labels.append("Correct")
		training_vectors.append(vector)
		
		
	for segment in collection_of_negative_segments.list_of_segments:
		vector = []
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
			vector.extend (segment.feature_dict["border_to_nucleus_distance_hist"])
			vector.append (segment.feature_dict["border_to_nucleus_distance_mean"])
			vector.append (segment.feature_dict["border_to_nucleus_distance_std"])
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_size"]):
			vector.append (segment.feature_dict["size"])
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_intensity"]):
			vector.append (segment.feature_dict["border_to_interior_intensity_ratio"])

		training_labels.append("Incorrect")
		training_vectors.append(vector)
		
	
	training_vectors = normalize_train_data(training_vectors)
	
		
	data = VectorDataSet(training_vectors,L=training_labels)
		

	return data
	
def prepare_SVM(data):

	data.attachKernel("gaussian")
	classif = SVM(optimizer="libsvm")
	
	classif.train(data)


	return classif



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



