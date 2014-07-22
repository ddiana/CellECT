# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import copy
from pygraph.classes.digraph import digraph
from pygraph.algorithms.accessibility import connected_components
import numpy as np
import re
#from openopt import LP
import pdb
import copy
from numpy import exp
import cv2

# Imports from this project
import CellECT.track_tool.globals
import CellECT.track_tool.cell_tracker_core.tracklet_connector as tc


"""
CellTracker class.
Takes list of cell profiles for every time stamp.
Creates distance matrix between nodes in consecutive time stamps.
Creates graph using these points and makes confident associations between
consecutive nodes.
"""


class CellTracker(object):

	def __init__(self):

		"""
		Empty Cell Tracker: empty list of cell profiles, empty graph and
		zero distance matrix
		"""

		self.list_of_cell_profiles_per_timestamp = []
		self.distance_matrix_list = []
		self.graph = digraph()
		self.max_time = 0



	def add_cell_profiles_per_timestamp(self, cell_profiles):
	
		"""
		Stores the cell profiles per time stamp and creates distance matrix for
		two consecutive time stamps.
		"""
			
		copy_cell_profiles = copy.deepcopy(cell_profiles)
		self.list_of_cell_profiles_per_timestamp.append(copy_cell_profiles)
	
		self.max_time = len(self.list_of_cell_profiles_per_timestamp)-1

		if len(self.list_of_cell_profiles_per_timestamp) > 1:
			t1 = len(self.list_of_cell_profiles_per_timestamp)-2
			t2 = len(self.list_of_cell_profiles_per_timestamp)-1
			self.distance_matrix_list.append(self.get_distance_matrix_for_t1_t2( t1, t2 ))

	def test_number_incoming_connections_equals_number_outgoing_connections(self):

		outgoing_dict = dict( map(lambda x,y: (x,y), self.ct.graph.nodes(), [0 for i in xrange(len(self.ct.graph.nodes()))]))

		incoming_dict = dict( map(lambda x,y: (x,y), self.ct.graph.nodes(), [0 for i in xrange(len(self.ct.graph.nodes()))]))

		for edge in pygraph.edges():

			outgoing_dict[edge[0]] += 1
			incoming_dict[edge[1]] += 1

		for node in outgoing_dict.keys():

			self.assertEqual( outgoing_dict[node] , incoming_dict[node])


	def test_one_incoming_track_per_node(self):		
			
			
	
		incoming_dict = dict( map(lambda x,y: (x,y), self.ct.graph.nodes(), [0 for i in xrange(len(self.ct.graph.nodes()))]))

		for edge in pygraph.edges():

			incoming_dict[edge[1]] += 1

		for node in outgoing_dict.keys():

			self.assertLessThanOrEqual( incoming_dict[node], 1)


	def test_one_incoming_track_per_node(self):		
			
		
		outgoing_dict = dict( map(lambda x,y: (x,y), self.ct.graph.nodes(), [0 for i in xrange(len(self.ct.graph.nodes()))]))

		for edge in pygraph.edges():

			outgoing_dict[edge[0]] += 1

		for node in outgoing_dict.keys():

			self.assertLessThanOrEqual( outgoing_dict[node], 1)



	def build_lineage(self):


		"""
		Build graph of cells. Do confident associations, and then connect resulting
		tracklets.
		"""

		self.add_nodes_to_graph()
		self.add_confident_associations_to_graph()	

#		tc.TrackletConnector(self)

		
	
		#self.create_program()


	

#	def connect_program():

#		numVar = len(gr.edges())

#		f = getCostsAndRewards(segList,gr, segProps, horizEdgeProps)

#		A = getTransitivityConstraints(gr)
#		b = np.ones(np.shape(A)[0])

#		lb = np.zeros(np.shape(f))
#		ub = np.ones(np.shape(f))
#	
#		print "LP"
#		p = LP(f, A=A, Aeq=[], b=b, beq=[], lb=lb, ub=ub)

#		print "Solving LP"
#		r = p.minimize('cvxopt_lp')



#		#writeLogFile(gr,f, r.xf, A)
#	
#		#plotWithProbe(segList[1],1,gr, A)

#		return r.xf, f
#				




	def add_nodes_to_graph(self):

		"""
		For every cell in the dataset add a node to the graph.
		Node is in format: "t%d_c%d" % (timestamp, cell_index)
		"""

		for i in xrange (0, len( self.list_of_cell_profiles_per_timestamp) ):
			for cp in xrange ( len(self.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles)):
				node_name = "t%d_c%d" % (i, cp)
				self.graph.add_node(node_name)


	def determine_histogram_max_bin(self, feat):


		feat_sum = 0
		for cp in self. list_of_cell_profiles_per_timestamp[0].list_of_cell_profiles:
			feat_sum += cp.dict_of_features[feat]	
		feat_avg = feat_sum / float(len(self.list_of_cell_profiles_per_timestamp[0].list_of_cell_profiles))

		max_val = feat_avg * 2

		return max_val

	def get_feature_hist_over_time(self, feat):


		num_bins = 25
		min_val = 0
		max_val = self. determine_histogram_max_bin(feat)


		for t in xrange(len(self.list_of_cell_profiles_per_timestamp)):
		
			cpt = self.list_of_cell_profiles_per_timestamp[t]


			vals_list = [x.dict_of_features[feat] for x in cpt.list_of_cell_profiles]
			hist = np.histogram( vals_list, num_bins, (min_val, max_val) )

			bins = hist[1]
			hist = hist[0] / float( np.sum(hist[0]))
			self.list_of_cell_profiles_per_timestamp[t].feature_histograms[feat] = (hist, bins)


	def get_feature_hist_over_time_2d(self, feature_pair):

		feat1 = feature_pair[0]
		feat2 = feature_pair[1]

		num_bins = 10
		min_val = 0
		feat_sum = 0
		max_val1 = self.determine_histogram_max_bin(feat1)
		max_val2 = self.determine_histogram_max_bin(feat2)



		for t in xrange(len(self.list_of_cell_profiles_per_timestamp)):
		
			cpt = self.list_of_cell_profiles_per_timestamp[t]

			vals_list1 = [x.dict_of_features[feat1] for x in cpt.list_of_cell_profiles]
			vals_list2 = [x.dict_of_features[feat2] for x in cpt.list_of_cell_profiles]

			hist = np.histogram2d( vals_list1, vals_list2, num_bins, [[min_val, max_val1], [min_val, max_val2]] )

			bins1 = hist[1]
			bins2 = hist[2]
			hist = hist[0] / float( np.sum(hist[0]))
			self.list_of_cell_profiles_per_timestamp[t].feature_histograms[(feat1, feat2)] = (hist, bins1, bins2)



	def compute_stats(self):

		self.get_feature_hist_over_time( "volume_by_res" )
		self.get_feature_hist_over_time( "flatness" )
		self.get_feature_hist_over_time( "elongation" )
		self.get_feature_hist_over_time( "sphericity" )
		self.get_feature_hist_over_time( "surface_area_by_res" )
		self.get_feature_hist_over_time( "entropy" )

		self.get_feature_hist_over_time_2d( ("centroid_dist_from_margin", "sphericity"))
		self.get_feature_hist_over_time_2d( ("centroid_dist_from_margin", "flatness"))
		self.get_feature_hist_over_time_2d( ("centroid_dist_from_margin", "elongation"))
		self.get_feature_hist_over_time_2d( ("centroid_dist_from_margin", "entropy"))
		self.get_feature_hist_over_time_2d( ("centroid_dist_from_margin", "volume_by_res"))
		self.get_feature_hist_over_time_2d( ("centroid_dist_from_margin", "surface_area_by_res"))



	def cost_for_pair(self, cp1, cp2):



		ct1 = np.reshape(np.array(cp1.dict_of_features["mid_slice_best_contour"]) , (len(cp1.dict_of_features["mid_slice_best_contour"]), 1, 2))
		ct2 = np.reshape(np.array(cp2.dict_of_features["mid_slice_best_contour"]) , (len(cp2.dict_of_features["mid_slice_best_contour"]), 1, 2))
		
		shape_cost = cv2.matchShapes(ct1, ct2, 2,0)

		print shape_cost
		return shape_cost


		xres = 0.3 # CellECT.track_tool.globals.PARAMETER_DICT["xres"]
		yres = 0.3 # CellECT.track_tool.globals.PARAMETER_DICT["yres"]
		zres = 2.0 #CellECT.track_tool.globals.PARAMETER_DICT["zres"]

		voxel_size = xres*yres*zres

		centroid_distance = (((cp1.nucleus.x - cp2.nucleus.x) * xres)**2 + ((cp1.nucleus.y - cp2.nucleus.y)*yres)**2 + ((cp1.nucleus.z - cp2.nucleus.z)*zres)**2) **0.5

		first_cell_size = cp1.size *voxel_size
		equivalent_sphere_radius = (3*first_cell_size / 4. / 3.14) ** (1/3.)

		sigma = equivalent_sphere_radius *2

	
		x = centroid_distance # / equivalent_sphere_radius * 100

		weight = 1 - exp(- (x**2) / (2* sigma **2)) #* 1./ (sigma * (2.*3.14)**0.5) 


		centroid_distance_score =  x*weight

		

		percent_size_difference = min( cp2.size / float(cp1.size) * float(voxel_size) ,cp1.size / float(cp2.size) * float(voxel_size)  )   # closer to 1 is better

		score_size = abs (1-percent_size_difference)  # small is better


		#centroid_distance_relative_to_size = centroid_distance / (0.5*cp1.size + 0.5*cp2.size)  # small is better

		score = centroid_distance_score * score_size

		print centroid_distance, equivalent_sphere_radius, x, weight, weight * x, score_size, score

		return centroid_distance_score


	def get_hungarian_association(self, cp, cp_list_1, cp_list_2):

		from munkres import Munkres

		if len(cp_list_2) ==0:
			return None

		cost_to_none = 25

		matrix = np.zeros((len(cp_list_1) +1, len(cp_list_1) +1 + len(cp_list_2))) + cost_to_none

		print matrix.shape

		for i in xrange(len(cp_list_1)):

			cp_i = cp_list_1[i]

#			count = 0
			for j in xrange(len(cp_list_2)):

#				count += 1
				cp_j = cp_list_2[j]
				
				matrix[i,j] = self.cost_for_pair(cp_i, cp_j)
#		print count


#		count = 0
		for j in xrange(len(cp_list_2)):
			cp_j = cp_list_2[j]
			matrix[-1,j] =  self.cost_for_pair(cp, cp_j)
#			count += 1
#		print count

		solver = Munkres()

		associations = solver.compute(matrix)

		result = None
		if associations[-1][1] < len(cp_list_2):
			result = associations[-1][1]

		return result

		
		


	def add_confident_associations_to_graph(self):

		threshold_dist = CellECT.track_tool.globals.PARAMETER_DICT["cell-association-distance-threshold"]
		threshold_cell_size_growth = CellECT.track_tool.globals.PARAMETER_DICT["max-cell-growth-rate"]



		# for every timestamp 
		for i in xrange (0, len( self.list_of_cell_profiles_per_timestamp) -1):

			matrix = self.distance_matrix_list[i]
			used_cells = np.zeros(len(self.list_of_cell_profiles_per_timestamp[i+1].list_of_cell_profiles))
			
			cell_profiles_this_timestamp = self.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles
			cell_profiles_next_timestamp = self.list_of_cell_profiles_per_timestamp[i+1].list_of_cell_profiles
	
			# for all cell_profiles in the first of two consecutive time_stamps
			for cp1 in self.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles:

				cp1_index = self.list_of_cell_profiles_per_timestamp[i].seg_label_to_cp_list_index[ cp1.label]

				node_name = "t" + str(i) + "_c" + str(cp1_index)

				bbx_group1 = self.list_of_cell_profiles_per_timestamp[i]. get_bounding_box_of_group(cp1.neighbor_labels)

				cp_list_labels_1 = [self.list_of_cell_profiles_per_timestamp[i].seg_label_to_cp_list_index[l] for l in cp1.neighbor_labels]

				cp_list_1 = [self.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles[j] for j in cp_list_labels_1]
				# remove the ones already associated
				cp_list_1 = filter(lambda x:  self.list_of_cell_profiles_per_timestamp[i].seg_label_to_cp_list_index[x.label] > cp1_index, cp_list_1)


				#bbx_group1.extend_by(30,  ( 800, 500, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"]))   )
				cp_list_2 = self.list_of_cell_profiles_per_timestamp[i+1].get_cells_within_space(bbx_group1)	
				# remove the ones already associated
				cp_list_2 = filter(lambda x: not used_cells[ self.list_of_cell_profiles_per_timestamp[i+1].seg_label_to_cp_list_index[x.label] ], cp_list_2)
				

				link_with = self.get_hungarian_association(cp1, cp_list_1, cp_list_2)

				if link_with is not None:

					cp_used = cp_list_2[link_with]
					cp_used_index = self.list_of_cell_profiles_per_timestamp[i+1].seg_label_to_cp_list_index[ cp_used.label]

					used_cells[cp_used_index] = 1
					node_name_closest = "t" + str(i+1) + "_c" + str(cp_used_index)
					self.graph.add_edge((node_name, node_name_closest))
					


		print "cells in 1:", len(self.list_of_cell_profiles_per_timestamp[0].list_of_cell_profiles)
		print "cells in 2:", len(self.list_of_cell_profiles_per_timestamp[1].list_of_cell_profiles)
				

#				node_name = "t" + str(i) + "_c" + str(cp1)

#				correct = False
#				num_tries = 0			
#			
#				

#				matrix_row = copy.deepcopy(matrix[cp1,:])


		



	def add_confident_associations_to_graph1(self):

		"""
		Associate nuclei from two consecutive time stamps based on how close
		together the points are to the ponts in the next time stamp.
		Also confirm that the size of the cells stays relatively constant.
		"""

		threshold_dist = CellECT.track_tool.globals.PARAMETER_DICT["cell-association-distance-threshold"]
		threshold_cell_size_growth = CellECT.track_tool.globals.PARAMETER_DICT["max-cell-growth-rate"]

		# for every timestamp 
		for i in xrange (0, len( self.list_of_cell_profiles_per_timestamp) -1):

			matrix = self.distance_matrix_list[i]
			used_cells = np.zeros((matrix.shape[1],1))
			
			cell_profiles_this_timestamp = self.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles
			cell_profiles_next_timestamp = self.list_of_cell_profiles_per_timestamp[i+1].list_of_cell_profiles
	
			# for all cell_profiles in the first of two consecutive time_stamps
			for cp1 in xrange( matrix.shape[0] ):

				node_name = "t" + str(i) + "_c" + str(cp1)

				correct = False
				num_tries = 0			
			
				matrix_row = copy.deepcopy(matrix[cp1,:])

				

				# try 5 times to find the closest node that matches
				while (not correct) and num_tries < 5:

					num_tries += 1
					
					closest = matrix_row.argmin()

					# if point has not yet been used 
					if used_cells[closest] == 0:

						# closest node has to be close enough
						if matrix[cp1,closest] < threshold_dist:

							# closest node has to have similar cell size
							
							try:
								if cell_profiles_this_timestamp[cp1].size < cell_profiles_next_timestamp[closest].size * threshold_cell_size_growth:
									# if we found it:
									correct = True
									used_cells[closest] = 1
									node_name_closest = "t" + str(i+1) + "_c" + str(closest)
									self.graph.add_edge((node_name, node_name_closest))
									break
							except:
								pdb.set_trace()
						else:
							break
							
					if not correct:
						matrix_row[closest] = 100000. # so that it doesnt get picked next
					
			

	def get_cell_profile_info_from_node_name(self,node_name):

		"""
		None names are e.g. t10_c239.
		Extract this information to get the time stamp and cell index.
		"""

		res = re.match("t(\d+)_c(\d+)", node_name)

		t = int(res.group(1))
		c = int(res.group(2))

		return t, c
	


		
	def get_distance_matrix_for_t1_t2(self,t1,t2):

		"""
		Euclidean distance between the nuclei of two consecutive time stamps.
		"""

		matrix = np.zeros((len(self.list_of_cell_profiles_per_timestamp[t1].list_of_cell_profiles), len(self.list_of_cell_profiles_per_timestamp[t2].list_of_cell_profiles)))

		counter1 = -1
		

		for cp1 in self.list_of_cell_profiles_per_timestamp[t1].list_of_cell_profiles:
			counter1 += 1
			counter2 = -1
			for cp2 in self.list_of_cell_profiles_per_timestamp[t2].list_of_cell_profiles:
				counter2 += 1
				try:
					matrix[counter1, counter2] = np.sqrt((cp1.nucleus.x - cp2.nucleus.x ) **2 + (cp1.nucleus.y - cp2.nucleus.y ) **2 +( float(CellECT.track_tool.globals.PARAMETER_DICT["z-scale"])* cp1.nucleus.z - float(CellECT.track_tool.globals.PARAMETER_DICT["z-scale"])* cp2.nucleus.z ) **2 )
				except:
					pdb.set_trace()

		return matrix
	

