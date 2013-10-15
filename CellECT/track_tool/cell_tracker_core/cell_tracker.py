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


	def add_cell_profiles_per_timestamp(self, cell_profiles):
	
		"""
		Stores the cell profiles per time stamp and creates distance matrix for
		two consecutive time stamps.
		"""
			
		copy_cell_profiles = copy.deepcopy(cell_profiles)
		self.list_of_cell_profiles_per_timestamp.append(copy_cell_profiles)
		if len(self.list_of_cell_profiles_per_timestamp) > 1:
			t1 = len(self.list_of_cell_profiles_per_timestamp)-2
			t2 = len(self.list_of_cell_profiles_per_timestamp)-1
			self.distance_matrix_list.append(self.get_distance_matrix_for_t1_t2( t1, t2 ))


	def build_lineage(self):


		"""
		Build graph of cells. Do confident associations, and then connect resulting
		tracklets.
		"""

		self.add_nodes_to_graph()
		self.add_confident_associations_to_graph()	

		#tc.TrackletConnector(self)

		
	
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



	def add_confident_associations_to_graph(self):

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
			
				matrix_row = matrix[cp1,:]

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
									used_cells[correct] = 1
									node_name_closest = "t" + str(i+1) + "_c" + str(closest)
									self.graph.add_edge((node_name, node_name_closest))
							except:
								pdb.set_trace()
							
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
	

