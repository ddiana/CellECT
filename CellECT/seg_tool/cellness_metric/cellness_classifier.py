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
from sklearn.cluster import MeanShift
from sklearn.neighbors import KNeighborsClassifier
from pygraph.algorithms.minmax import shortest_path
from sklearn.semi_supervised import label_propagation
from pygraph.classes.graph import graph

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


		self.positive_examples = []
		self.negative_examples = []

		self.mean_per_dim_specific = []
		self.std_per_dim_specific = []
		self.mean_per_dim_generic = []
		self.std_per_dim_generic = []
	
		self.hist_of_feature = {}
		self.bins_of_feature = {}

		self.data_pos = None
		self.data_neg = None

		self.feature_set = []

		self.generic_classifier = None
		self.specific_classifier = None

		self.feedback_correct_score = None

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


	def determine_histogram_max_bin(self, feat, segment_collection):


		feat_sum = 0
		for seg in segment_collection.list_of_segments:
			feat_sum += seg.feature_dict[feat]	
		feat_avg = feat_sum / float(len(segment_collection.list_of_segments))

		max_val = feat_avg * 2

		return max_val

	def get_spatial_histogram(self, feature_set, segment_collection):


		num_bins = 5
		min_val = 0
		feat_sum = 0


		#max_val_AP_dist = self.determine_histogram_max_bin("dist_to_AP_axis", segment_collection)
		#max_val_AP_pos = self.determine_histogram_max_bin("position_along_AP_axis", segment_collection)
		#max_val_dist_margin = self. determine_histogram_max_bin("centroid_dist_from_margin", segment_collection)

		#max_val_feat = [ self. determine_histogram_max_bin(feat, segment_collection) for feat in feature_set]


		vals_ap_pos = [x.feature_dict["position_along_AP_axis"] for x in segment_collection.list_of_segments]
		vals_ap_dist = [x.feature_dict["dist_to_AP_axis"] for x in segment_collection.list_of_segments]
		vals_dist_margin = [x.feature_dict["centroid_dist_from_margin"] for x in segment_collection.list_of_segments]
		
		counter = 0

		for feat in feature_set:
			vals = [x.feature_dict[feat] for x in segment_collection.list_of_segments]

			vals_vector = np.array(zip ([vals, vals_ap_pos, vals_ap_dist, vals_dist_margin])).squeeze().T

			

			hist, bins = np.histogramdd( vals_vector, bins = [num_bins, num_bins, num_bins, num_bins]) #,   range =[ min_val, max_val_feat, min_val, max_val_AP_pos, min_val, max_val_AP_dist, min_val, max_val_dist_margin] )


			self.hist_of_feature[feat] = hist / float (hist.max())
			self.bins_of_feature[feat] = bins[0]
			
			if counter ==0:

				self.bins_of_feature["position_along_AP_axis"]= bins[1]
				self.bins_of_feature["dist_to_AP_axis"] =bins[2]
				self.bins_of_feature["centroid_dist_from_margin"] = bins[3]
			counter += 1			
		


	
	def get_cluster_centers(self, segment_collection):

		self. normalized_features  = np.zeros((len(self.feature_set), len(segment_collection.list_of_segments)))
		
	
		counter = 0
		for feat in self.feature_set:

			vals = np.array([x.feature_dict[feat] for x in segment_collection.list_of_segments])
			self.normalized_features[counter,:] = (vals - vals.mean()	)/ vals.std()

			counter += 1

		ms = MeanShift()
		ms.fit(self.normalized_features.T)
		
		self.cluster_labels = ms.labels_
		self.cluster_centers = ms.cluster_centers_


		self.knn = KNeighborsClassifier()
		self.knn.fit (self.normalized_features.T, self.cluster_labels)		



	def get_dissimilarity_to_neighbors(self, segment_collection):

		# HIGH VALUES FOR GOOD SEGMENTS
		# between 0 and 1
	
		num_segs = len(segment_collection.list_of_segments)
		self. neighbor_dissimilarity = np.zeros((num_segs, num_segs))



		for i in xrange( num_segs ):

			seg = segment_collection.list_of_segments[i]
			neighbors = [x[0] for x in seg.feature_dict["mean_intensity_border_with_neighbor"]]

			for n_label in neighbors:
				n = segment_collection.segment_label_to_list_index_dict[n_label]
				if self.neighbor_dissimilarity[i,n] ==0:
					val = self. get_dissimilarity(i, n)
					self.neighbor_dissimilarity[i,n] = val
					self.neighbor_dissimilarity[n,i] = val


		self.avg_neighbor_dissimilarity = []
		for i in xrange(num_segs):
			val = np.mean(filter(lambda x: x>0, self. neighbor_dissimilarity[i,:]))
			if np.isnan(val):
				val = 0
			self.avg_neighbor_dissimilarity .append(val)
		

		self.avg_neighbor_dissimilarity = 1- np.array(self.avg_neighbor_dissimilarity ) / max(self.avg_neighbor_dissimilarity)




	def get_dissimilarity(self, idx1, idx2):

		
		feature_dist = np.linalg.norm(self.normalized_features[:,idx1] - self.normalized_features[:,idx2])

		idx1_probab = self.knn.predict_proba(self.normalized_features[:,idx1])[0]
		idx2_probab = self.knn.predict_proba(self.normalized_features[:,idx2])[0]


		# expected cluster distance
		cluster_dist = 0 
		if len(idx1_probab)>1:
			for i in xrange(len(idx2_probab)-1):
				for j in xrange(i+1, len(idx2_probab)):
					cluster_dist = idx1_probab[j] * idx2_probab[i] * np.linalg.norm( self.cluster_centers[i] - self. cluster_centers[j])

		cluster_dist = 0

		return feature_dist + cluster_dist
		
	def select_features(self, segment_collection):

		if len(self.feature_set)>0:
			return

		all_features = ["volume_by_res", "flatness", "elongation", "sphericity", "cylindricity", "entropy", "vol_by_res_to_enclosing_box_vol_ratio"]
		self.get_spatial_histogram(all_features, segment_collection)

		self.entropy_of_histogram = {}	

		for feat in all_features:

			self.entropy_of_histogram[feat] = - sum((x*np.log(x) for x in filter(lambda y: y>0,self.hist_of_feature[feat].flatten())))	

		thresh = np.median(self.entropy_of_histogram.values())

		

		for feat in self.entropy_of_histogram.keys():
			if self.entropy_of_histogram[feat] <= thresh:
				self.feature_set.append(feat)

		print "Using features:", self.feature_set

	def apply_metric(self, segment_collection = None, correct_labels=None): #, correct_labels = [], incorrect_labels = []):

		if segment_collection is not None:
			self.select_features(segment_collection)
			
			self.get_cluster_centers(segment_collection)
			self.get_dissimilarity_to_neighbors(segment_collection)
			self.get_intrinsic_scores(segment_collection)


			self.feedback_score(segment_collection, correct_labels)

#			#self.get_model_scores(segment_collection, feature_set)


			self.propagate_labels(segment_collection)


	def add_examples_by_label(self, segment_collection, positive_labels, negative_labels):		
		

		idx_pos = [segment_collection.segment_label_to_list_index_dict[x] for x in positive_labels]
		idx_neg = [segment_collection.segment_label_to_list_index_dict[x] for x in negative_labels]
		
		
		data_pos = np.vstack([[self.convexity[i], self.avg_border_with_neighbors_scores[i], self.border_to_interior[i], self.avg_neighbor_dissimilarity[i]] for i in idx_pos])
		data_neg = np.vstack([[self.convexity[i], self.avg_border_with_neighbors_scores[i], self.border_to_interior[i], self.avg_neighbor_dissimilarity[i]] for i in idx_neg])

		if self.data_pos is not None and data_pos is not None:
			self.data_pos = np.vstack([self.data_pos, data_pos])

		else:
			self.data_pos = data_pos

		if self.data_neg is not None and data_neg is not None:
			self.data_neg = np.vstack([self.data_neg, data_neg])
		else:
			self.data_neg = data_neg


	def propagate_labels(self, segment_collection): #, positive_labels, negative_labels):

#		if len (positive_labels) == 0 or len(negative_examples==0):
#			self.label_propagation_score = [0] * len(segment_collection)
#			self.predictor_validity = 0
#			return

#		idx_pos = [segment_collection.segment_label_to_list_index_dict[x] for x in positive_labels]
#		idx_neg = [segment_collection.segment_label_to_list_index_dict[x] for x in negative_labels]
#
		num_pos = 0
		if self.data_pos is not None:
			num_pos = len(self.data_pos)

		num_neg = 0
		if self.data_neg is not None:
			num_neg = len(self.data_neg)

		if num_pos == 0 or num_neg == 0:
			self.label_propagation_score = [0] * len(segment_collection.list_of_segments)
			self.predictor_validity = 0	
			return
		

		new_data = np.vstack([self.convexity, self.avg_border_with_neighbors_scores, self.border_to_interior, self.avg_neighbor_dissimilarity]).T
	
		all_data = new_data
		if self.data_pos is not None:
			all_data = np.vstack([new_data,self.data_pos])

		if self.data_neg is not None:
			all_data = np.vstack([all_data, self.data_neg])
	
		labels = np.ones((all_data.shape[0]))* -1
		labels[range(self.data_pos.shape[0])] = 1
		labels[range(self.data_pos.shape[0],self.data_neg.shape[0]+ self.data_pos.shape[0])] = 0			

		predictor = label_propagation.LabelSpreading().fit(all_data,labels)


		self.predictor_validity = (2*min (num_pos, num_neg) / float (num_pos + num_neg)) * ((1./ (1+ np.exp(-0.1 * (num_pos + num_neg))) - 0.5) *2)

		self.label_propagation_score = [predictor.label_distributions_[i][0] for i in xrange (len(labels))]


	def feedback_score(self, segment_collection, correct_labels):

		# HIGH VALUES FOR GOOD
		# 0 to 1

		try:
			self.build_graph(segment_collection)
			min_dist_all = {}
			for label in correct_labels:

				min_dist =  shortest_path(self.graph, label)[1]

				if len(min_dist_all.keys()) >0:

					for x in min_dist.keys():
						min_dist_all[x] = max ((min_dist_all[x], np.exp(-min_dist[x])))
				else:
					
					for x in min_dist.keys():
						min_dist_all[x] = np.exp( -min_dist[x])

			self.feedback_correct_score = [0] * len(segment_collection.list_of_segments)

	#		hist, bins = np.histogram(self.avg_neighbor_dissimilarity)
	#		max_loc = np.argmax(hist)
	#		sigma = 10 # (bins[max_loc] + bins[max_loc+1]) / 2. 


			if len(correct_labels)>0:
				max_val = max (min_dist_all.values())
				for label in min_dist_all.keys():
					seg_idx = segment_collection.segment_label_to_list_index_dict[label]

					#new_val = np.exp(-min_dist_all[label]**2./(2.*sigma**2.))
					self.feedback_correct_score[seg_idx] = min_dist_all[label]
		
		except Exception as err:
			pdb.set_trace()
			

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

	

	def avg_border_with_neighbor(self, segment):

		# changed mean to min		

		vals = [x[1] for x in segment.feature_dict["mean_intensity_border_with_neighbor"]]


		boundary_intensity_score = 0
		if len(vals):

			boundary_intensity_score = np.min(vals)
		else:
			vals = segment.feature_dict["border_intensity_hist"]
			boundary_intensity_score = sum([vals[i] * (i+ 0.5)/20.* 255 for i in xrange(20)])

		#boundary_to_interior = segment.feature_dict["border_to_interior_intensity_hist_dif"]

		return boundary_intensity_score

		
	


	def build_graph(self, segment_collection):

		
		self.graph = graph()

		rescaled_neighbor_dissimilarity = self.neighbor_dissimilarity / float (self.neighbor_dissimilarity.max())

		self.graph.add_nodes([x.label for x in segment_collection.list_of_segments])

		for segment in segment_collection.list_of_segments:
			x = segment.label
	
			x_idx = segment_collection.segment_label_to_list_index_dict[x]

			for y in (tup[0] for tup in segment.feature_dict["mean_intensity_border_with_neighbor"]):

				if not self.graph.has_edge((x, y)):

					y_idx = segment_collection.segment_label_to_list_index_dict[y]
	
					weight = -np.log((1- rescaled_neighbor_dissimilarity[x_idx, y_idx]))

					self.graph.add_edge((x,y), weight )

		

	def get_intrinsic_scores(self, segment_collection):
		
		# HIGH VALUES FOR GOOD SEGMENTS
		# between 0 and 1
	
		self.avg_border_with_neighbors_scores = [0] * len(segment_collection.list_of_segments)
		self.border_to_interior = [0] * len(segment_collection.list_of_segments)
		self.convexity =  [0] * len(segment_collection.list_of_segments)

		for i in xrange(len(segment_collection.list_of_segments)):


			segment = segment_collection.list_of_segments[i]
			self.avg_border_with_neighbors_scores[i] = self.avg_border_with_neighbor(segment)
			self.border_to_interior[i] = segment.feature_dict["border_to_interior_intensity_hist_dif"]

			self.convexity[i] = segment.feature_dict["vol_to_hull_vol_ratio"]

		self.avg_border_with_neighbors_scores = np.array(self.avg_border_with_neighbors_scores) / 255.
		self.border_to_interior = 1 - np.array(self.border_to_interior) / max(self.border_to_interior)

		


	def find_bin_index(self, bins, value):

		idx = 0
		while value > [idx]:
			idx += 1
		idx -=1
		if idx <0:
			idx =0
		return idx
		

	def get_model_scores(self, segment_collection):

		pdb.set_trace()

#		self.model_prob_per_feature = {}

#		num_segs = len (segment_collection.list_of_segments)

#		for feat in feature_set:
#			self.model_prob_per_feature[feat] = [0] * num_segs

#		
#		for i in xrange(num_segs):
#			segment = segment_collection.list_of_segments[i]

#			ap_dist = segment.feature_dict["dist_to_AP_axis"]
#			ap_pos = segment.feature_dict["position_along_AP_axis"]
#			margin_dist = segment. feature_dict["centroid_dist_from_margin"]

#			min_idx_ap_pos = self.find_bin_index(self.bins_of_feature["position_along_AP_axis"], ap_pos - 10)
#			min_idx_ap_dist = self.find_bin_index(self.bins_of_feature["position_along_AP_axis"], ap_dist - diameter)
#			min_idx_margin_dist = self.find_bin_index(self.self.bins_of_feature["centroid_dist_from_margin"], margin_dist - diameter)
#			

#			max_idx_ap_pos = self.find_bin_index(self.bins_of_feature["position_along_AP_axis"], ap_pos + 10)
#			max_idx_ap_dist = self.find_bin_index(self.bins_of_feature["position_along_AP_axis"], ap_dist + diameter)
#			max_idx_margin_dist = self.find_bin_index(self.self.bins_of_feature["centroid_dist_from_margin"], margin_dist + diameter)
#			


#			for feat in feature_set:

#				min_idx_feat = self.find_bin_index(self.self.bins_of_feature[feat], feat * 0.8)
#				max_idx_feat = 	self.find_bin_index(self.self.bins_of_feature[feat], feat * 1.2)

#				hist = self. hist_of_feature[feat]
#				common_cube = hist[min_idx_feat: max_idx_feat+1, min_idx_ap_pos: max_idx_ap_pos, min_idx_ap_dist:max_idx_ap_dist, min_idx_margin_dist:max_idx_margin_dist].sum()
#				strip1 = hist[:, min_idx_ap_pos: max_idx_ap_pos, min_idx_ap_dist:max_idx_ap_dist, min_idx_margin_dist:max_idx_margin_dist].sum()


#				prob = common_cube / (strip1 + strip2 + strip2 -2 * common_cube)

#			idx_ap_dist = 0
#			while ap_dist > self.bins_of_feature["dist_to_AP_axis"][idx_ap_dist]:
#				idx_ap_dist += 1
#			idx_ap_dist -= 1
#			
#			idx_margin_dist = 0
#			while margin_dist > self.bins_of_feature["centroid_dist_from_margin"][idx_margin_dist]:
#				idx_margin_dist += 1
#			idx_margin_dist -=1

#		
#			diameter = segment.feature_dict["minimum_enclosing_sphere_radius_by_res"] * 2
#	
#			min_ap_pos = (ap_pos - diameter)



#				feat_val = segment.feature_dict[feat]

#				idx_feat = 0
#				while feat_val > self.bins_of_feature[feat][idx_feat]:
#					idx_feat += 1
#				idx_feat -=1


#				prob = self. hist_of_feature[feat][ idx_feat,idx_ap_pos, idx_ap_dist, idx_margin_dist]

#				self.model_prob_per_feature[feat][i] = prob

		pdb.set_trace()

		
	def test_segment(self, seg_idx):

		#score =  #self.label_propagation_score[seg_idx] #self.feedback_correct_score[seg_idx] # self.avg_neighbor_dissimilarity[seg_idx] + self. border_to_interior[seg_idx] + self. convexity[seg_idx ] + self. avg_border_with_neighbors_scores[seg_idx]
		
		score = 0
#		score += self.avg_neighbor_dissimilarity[seg_idx]
#		score += self.avg_border_with_neighbors_scores[seg_idx]
#		score += self.border_to_interior[seg_idx]
#		score += self.convexity[seg_idx]
		score += self.feedback_correct_score[seg_idx]

		score /= 5.

		score = 1-score

		final_score = self.predictor_validity * self.label_propagation_score[seg_idx] + (1-self.predictor_validity) * score
		print final_score
				
		#self.model_prob_per_feature["flatness"][seg_idx]

		return "Correct", final_score
	
	def test_segment1(self, segment):

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



