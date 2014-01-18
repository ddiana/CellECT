import numpy as np
#from openopt import LP
import pygraph.algorithms.accessibility as algos
import pdb
import re


from CellECT.track_tool import cell_tracker_core
import CellECT.track_tool.globals


class TrackletConnectorOptProgram(object):

	def __init__(self, f, A,b):

		self.f = f
		self.A = A
		self.b = b
		
		self.lb = np.zeros(self.A.shape[1])
		self.ub = np.ones(self.A.shape[1])

		print "Solving LP"


	

		#self.p = LP(self.f, A=self.A, Aeq=[], b=self.b, beq=[], lb=self.lb, ub=self.ub)

		#self.r = self.p.maximize('linprog')

		pdb.set_trace()




class Tracklet(object):

	def __init__ (self,index):
		self.nodes = []
		self.first_node = None
		self.first_time_point = np.inf
		self.last_node = None
		self.last_time_point = -1
		self.index = index

		self.future_neighbors = []
		self.past_neighbors = []


	def add_node(self,node):

		self.nodes.append(node)
		t,c = self.split_node_name(node)

		if t < self.first_time_point:
			self.first_time_point = t
			self.first_cell = c
			self.first_node = node

		if t > self.last_time_point:
			self.last_time_point = t
			self.last_node = node
			self.last_cell = c

		

	def split_node_name(self,node_name):

		"""
		None names are e.g. t10_c239.
		Extract this information to get the time stamp and cell index.
		"""

		res = re.match("t(\d+)_c(\d+)", node_name)

		t = int(res.group(1))
		c = int(res.group(2))

		return t, c


		

class TrackletConnector(object):

	def __init__(self, ct):

		"""
		Get cell tracker. Extract info from graph and connect tracklets with LP.
		Update links in graph based on LP result.
		"""
		
		self.ct = ct
		self.get_tracklet_LP_variables()



	def get_tracklet_LP_variables(self):

		"""
		Get tracklet variables and their connection rewards
		"""
	
		# get connected components (tracklets)
		
		self.extract_tracklets()

		self.prep_tracklet_connections()

		#self.make_coexistence_matrix()
		self.make_weights()
		self.make_constraints_matrix()

		self.opt_result = TrackletConnectorOptProgram(self.costs, self.constraints_matrix, self.constraints_vector).r.xf

		self.add_links_to_graph()


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

		for edge in graph.edges():

			incoming_dict[edge[1]] += 1

		for node in outgoing_dict.keys():

			self.assertLessThanOrEqual( incoming_dict[node], 1)


	def test_one_incoming_track_per_node(self):		
			
		
		outgoing_dict = dict( map(lambda x,y: (x,y), self.ct.graph.nodes(), [0 for i in xrange(len(self.ct.graph.nodes()))]))

		for edge in graph.edges():

			outgoing_dict[edge[0]] += 1

		for node in outgoing_dict.keys():

			self.assertLessThanOrEqual( outgoing_dict[node], 1)


	def add_links_to_graph(self):

		"""
		Add links in the original graph, according to the result of the optimization program.
		"""


		for tracklet_connection_index in xrange(len(self.opt_result)):

			if self.opt_result[tracklet_connection_index]:
			
				# key might not be there because that's a link from source or sink
				if self.tracklet_connection_reverse_index.has_key(tracklet_connection_index):

					tracklet1_idx, tracklet2_idx = self.tracklet_connection_reverse_index[tracklet_connection_index]

					tracklet1 = self.tracklets[tracklet1_idx]
					tracklet2 = self.tracklets[tracklet2_idx]
				
					if not self.ct.graph.has_edge((tracklet1.last_node, tracklet2.first_node)):
						self.ct.graph.add_edge((tracklet1.last_node, tracklet2.first_node))

						print "Adding edge (%s, %s)" % (tracklet1.last_node, tracklet2.first_node)



	def make_weights(self):

		self.costs = np.zeros(len(self.tracklet_connections.keys()) + len(self.tracklet_connections_from_source)+ len(self.tracklet_connections_to_sink))

		# tracklet connections within the graph have costs	
		for tracklet in self.tracklet_connections.keys():
			tracklet_index, cost = self.tracklet_connections[tracklet]
			self.costs[tracklet_index] = cost


		# tracklet connections to/from sink have cost 0

		

	def prep_tracklet_connections(self): 

		"""
		From the list of tracklets get the variables for the LP,
		meaning the possible tracklet connections
		"""

		self.tracklet_connections = {}

		maximum_delta_t = 2
		minimum_tracklet_length = 1
		maximum_spacial_distance = 200

		self.links_out_of_tracklet = {}
		self.links_into_tracklet = {}

		self.tracklet_connection_reverse_index = {}


		def add_tracklet_connection(tracklet1, tracklet2, index):
			weight = self.get_tracklet_connection_weight(tracklet1.index, tracklet2.index)
			if weight < maximum_spacial_distance:

				if not self.tracklet_connections.has_key((tracklet1.index, tracklet2.index)):
					self.tracklet_connections[(tracklet1.index, tracklet2.index)] = (index[0], weight)

					self.tracklet_connection_reverse_index[index[0]] = (tracklet1.index, tracklet2.index)
		
					# add link into tracket2
					if not self.links_into_tracklet.has_key(tracklet2.index):
						self.links_into_tracklet[tracklet2.index] = set()
					self.links_into_tracklet[tracklet2.index].add(tracklet1.index)

					# add link out of tracket1
					if not self.links_out_of_tracklet.has_key(tracklet1.index):
						self.links_out_of_tracklet[tracklet1.index] = set()
					self.links_out_of_tracklet[tracklet1.index].add(tracklet2.index)

					index[0] += 1

		def add_source_sink_links(tracklet, index):
				
				

				if tracklet.split_node_name(tracklet.first_node)[0] == 0:

	
					self.links_out_of_source.add(index[0])
					self.tracklet_connections_from_source[tracklet.index] = (index[0], 0)
					index[0] += 1


				if tracklet.split_node_name(tracklet.last_node)[0] == self.ct.max_time:
			
					self.links_into_sink.add(index[0])
					self.tracklet_connections_to_sink[tracklet.index] = (index[0], 0)
					index[0] += 1

				


		tracklet_keys = self.tracklets.keys()
		
		self.tracklet_connections_from_source = {}
		self.tracklet_connections_to_sink = {}
		self.links_out_of_source = set()
		self.links_into_sink = set()
	


		index = [0]
		for i in xrange(len(tracklet_keys)-1):
			for j in xrange (i+1,len(tracklet_keys)):

				tracklet1 = self.tracklets[tracklet_keys[i]]
				tracklet2 = self.tracklets[tracklet_keys[j]]
				

				if len(tracklet1.nodes) > minimum_tracklet_length and len(tracklet2.nodes) > minimum_tracklet_length:

					if (tracklet2.first_time_point - tracklet1.last_time_point < maximum_delta_t) and (tracklet1.last_time_point < tracklet2.first_time_point):
						add_tracklet_connection(tracklet1, tracklet2, index )
			
					if (tracklet1.first_time_point - tracklet2.last_time_point < maximum_delta_t) and (tracklet2.last_time_point < tracklet1.first_time_point):
						add_tracklet_connection(tracklet2, tracklet1, index)


		for tracklet in self.tracklets.values():

			add_source_sink_links(tracklet, index)







	def make_constraints_matrix(self):


		number_vars = len(self.tracklet_connections) + len(self.tracklet_connections_from_source.keys()) + len(self.tracklet_connections_to_sink.keys())

		# constraints for sum of all incoming connections and sum of all outgoing connections (per node)
		# and one constraint for sum of sourse links = sum sink links
		number_constraints = len(self.tracklets) + 1 + len(self.tracklets)*3

		self.constraints_matrix = np.zeros((number_constraints, number_vars))

		self.constraints_vector = np.zeros((number_constraints,1))


		# sum of all incoming connections = sum of all outgoing connections
		# for every node: +1 for all incoming connections and -1 for all outgoing connections

		constraint_index = 0


		for trk in self.tracklets.keys():

			# all the tracklet connections ending in trk
			if self.links_into_tracklet.has_key(trk):
				for trk_in in self.links_into_tracklet[trk]:
					index, dist = self.tracklet_connections[(trk_in, trk)]
					self.constraints_matrix[constraint_index, index] = 1
			# is there also one coming from source?
			if self.tracklet_connections_from_source.has_key(trk):
				index = self.tracklet_connections_from_source[trk]
				self.constraints_matrix[constraint_index, index] = 1

			# all the tracklet connections that start at trk
			if self.links_out_of_tracklet.has_key(trk):
				for trk_out in self.links_out_of_tracklet[trk]:
					index, dist = self.tracklet_connections[(trk, trk_out)]
					self.constraints_matrix[constraint_index, index] = -1
			# is there also one coming from source?
			if self.tracklet_connections_to_sink.has_key(trk):
				index = self.tracklet_connections_to_sink[trk]
				self.constraints_matrix[constraint_index, index] = -1
				
			constraint_index += 1

		# sum of all connections out of source = sum of all connections into sink

		for trk in self.tracklet_connections_from_source.keys():
			index = self.tracklet_connections_from_source[trk]
			self.constraints_matrix[constraint_index, index] = 1
								
		for trk in self.tracklet_connections_to_sink.keys():
			index = self.tracklet_connections_to_sink[trk]
			self.constraints_matrix[constraint_index, index] = -1

		constraint_index += 1

		

		# sum of all connections out of a node == sum of all connections into that node

		for trk in self.tracklets.values():

			if self.links_into_tracklet.has_key(trk.index):

				for trk_into in self.links_into_tracklet[trk.index]:
					index, dist = self.tracklet_connections[(trk_into, trk.index)]
					self.constraints_matrix[constraint_index, index] = 1
			
			if self.links_out_of_tracklet.has_key(trk.index):

				for trk_out in self.links_out_of_tracklet[trk.index]:
					index, dist = self.tracklet_connections[(trk.index, trk_out)]
					self.constraints_matrix[constraint_index, index] = -1

			constraint_index += 1

		# sum of all connections out of a node <= 1

		for trk in self.tracklets.values():

			if self.links_into_tracklet.has_key(trk.index):

				for trk_into in self.links_into_tracklet[trk.index]:
					index, dist = self.tracklet_connections[(trk_into, trk.index)]
					self.constraints_matrix[constraint_index, index] = 1
			
				self.constraints_vector[constraint_index,0] = 1

				constraint_index += 1

		# sum of all connections into a node <= 1

		for trk in self.tracklets.values():

			if self.links_out_of_tracklet.has_key(trk.index):

				for trk_out in self.links_out_of_tracklet[trk.index]:
					index, dist = self.tracklet_connections[(trk.index, trk_out)]
					self.constraints_matrix[constraint_index, index] = 1

				self.constraints_vector[constraint_index,0] = 1

				constraint_index += 1



#	def make_connections_matrix(self):

#		"""
#		For every tracklet connection to every other tracklet connection:
#		-> -1 if they share the end tracklet
#		-> +1 if they share the begin tracklet 
#		-> 0 if they are not connected
#		"""


#		num_vars = len(self.tracklet_connections.keys()) + len (self.tracklet_connections_from_source) + len(self.tracklet_connections_to_sink)
#	
#		num_nodes = len(self.tc.)
#		num_constraints = num_nodes +  + len (self.tracklet_connections_from_source) + len(self.tracklet_connections_to_sink)
#		

#		self.connections_matrix = np.zeros((num_constraints, num_vars))

#	
#		for tc in self.tracklet_connections.keys()

#			index, dist = self.tracklet_connections[tc]
#			tracklet1, tracklet2 = tc 

#			# if tracklet1 is connected to s whats the index of that connection
#			index_source1 = -1
#			if tracklet1 in self.links_out_of_source:
#				index_source1, dist = self.tracklet_connections_from_source(tracklet1)

#			# if tracklet2 is connected to source whats the index of that connection
#			index_source2 = -1
#			if tracklet2 in self.links_out_of_source:
#				index_source2, dist = self.tracklet_connections_from_source(tracklet2)

#			# if tracklet1 is connected to the sink what's the index of that connection
#			index_sink1 = -1
#			if tracklet1 in self.links_into_sink:
#				index_sink1, dist = self.tracklet_connections_to_sink(tracklet1)
#		
#			# if tracklet2 is connected to the sink what's the index of that connection
#			index_sink2 = -1
#			if tracklet2 in self.links_into_sink:
#				index_sink2, dist = self.tracklet_connections_to_sink(tracklet2)


#			# add all the connections which are coming out of tracklet1
#			for trk in self.links_out_of_tracklet[tracklet1]:
#				tc_index = self.tracklet_connections[(tracklet1,trk)][0]
#				self.coexistence_matrix[current_tc_index, tc_index] = 1
#				self.coexistence_matrix[tc_index, current_tc_index] = 1



#			# find other tracklet_connections that end in tracklet2
#			for in_tracklet in self.links_into_tracklet(tracklet2):
#				index_in, dist = self.tracklet_connections[(in_tracklet, tracklet2)]
#				self.connections_matrix[index, index_in] = -1
#				# is tracklet2 also connected to source
#				if index_source2 > 0:
#					self.coexistence_matrix[index, index_source2] = -1


#			# find other tracklet_connections that start in tracklet1
#			for out_tracklet in self.links_out_of_tracklet(tracklet1):
#				index_out, dist = self.tracklet_connections[(tracklet1, out_tracklet)]
#				self.connections_matrix[index, index_out] = 1
#				# is tracklet1 also connected to sink
#				if index_sink1 > 0:
#					self.coexistence_matrix[index, index_sink2] = 1




#	def make_coexistence_matrix(self):

#		"""
#		Make coexistance matrix, which indicates which combinations of tracklets 
#		is possible. 
#		For every tracklet connection (connection which links tracket 1 and tracklet2)
#		- all the connections that start from tracklet1 = 1
#		- all the connections that end in tracklet2 = infinity 
#		- everythign else 0.
#		This means that two lineage lines cannot connect back, but they can split
#		"""

#	

#		for tracklet_connection in self.tracklet_connections.keys():

#			current_tc_index = self.tracklet_connections[tracklet_connection][0]
#			tracklet1 = tracklet_connection[0]
#			tracklet2 = tracklet_connection[1]

#			
#			for trk in self.links_out_of_tracklet[tracklet1]:
#				tc_index = self.tracklet_connections[(tracklet1,trk)][0]
#				self.coexistence_matrix[current_tc_index, tc_index] = 1
#				self.coexistence_matrix[tc_index, current_tc_index] = 1

#			for trk in self.links_into_tracklet[tracklet2]:
#				tc_index = self.tracklet_connections[(trk, tracklet2)][0]
#				self.coexistence_matrix[current_tc_index, tc_index] = 100000000000
#				self.coexistence_matrix[tc_index, current_tc_index] = 100000000000


	


	def get_tracklet_connection_weight(self,t1, t2):

		"""
		Given two tracklets, get the connection weight for them.
		Meaning how likely is it that these two should be connected.
		"""

		# check how for last node from t1 is from first node from t2
		t1_last_node = self.tracklets[t1].last_node
		t2_first_node = self.tracklets[t2].first_node

		t1_last_time = 	self.tracklets[t1].last_time_point
		t2_first_time = self.tracklets[t2].first_time_point

		t1_last_cell = 	self.tracklets[t1].last_cell
		t2_first_cell = self.tracklets[t2].first_cell

		
		t1_last_cell_profile = self.ct.list_of_cell_profiles_per_timestamp[t1_last_time].list_of_cell_profiles[t1_last_cell]
		t2_first_cell_profile = self.ct.list_of_cell_profiles_per_timestamp[t2_first_time].list_of_cell_profiles[t2_first_cell]
		
		cp1 = t1_last_cell_profile
		cp2 = t2_first_cell_profile
		
		dist = np.sqrt((cp1.nucleus.x - cp2.nucleus.x ) **2 + (cp1.nucleus.y - cp2.nucleus.y ) **2 +( float(CellECT.track_tool.globals.PARAMETER_DICT["z-scale"])* cp1.nucleus.z - float(CellECT.track_tool.globals.PARAMETER_DICT["z-scale"])* cp2.nucleus.z ) **2 )
		prob = 1 - np.double(dist)/float(CellECT.track_tool.globals.PARAMETER_DICT["cell-association-distance-threshold"])

		if prob == 0:
			prob = 0.01

		weight = np.log(prob / (1-prob))



		return weight



	def extract_tracklets(self):

		"""
		Make list of tracklet objects.
		"""

	
		self.cc = algos.connected_components(self.ct.graph)

		self.tracklets = dict(zip(self.cc.values(), [Tracklet(i) for i in self.cc.values()]))

		for node in self.cc.keys():
			self.tracklets[self.cc[node]].add_node(node)


		#self.tracklet_index_to_list_index = dict(zip([self.tracklets[t].index for t in self.tracklets.keys()], [i for i in xrange(len(self.tracklets))]))
		

		
