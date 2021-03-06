# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import scipy as sp
from scipy import io
import pdb
import time

# Imports from this project
from CellECT.seg_tool.seg_utils import union_find
from CellECT.seg_tool.features import segment_features as feat
from CellECT.seg_tool.seg_utils import voxel



class Nucleus(object):
	"Nucleus object. Comes either from nuclei detector, or added by user."	

	def __init__(self, x, y, z, index, is_added_by_user = False): #, t, confidence, index):

		self.x = x
		self.y = y
		self.z = z
#		self.t = t
#		self.confidence = confidence
		self.index = index
		self.added_by_user = is_added_by_user





class NucleusCollection(object):

	"Collection of nuclei and tools to manipulate it (add, union, etc)."

	def __init__(self, mat_path = None, mat_var = None):

		self.union_find = union_find.UnionFind(0)

		if mat_path and mat_var:
			self.__read_nuclei(mat_path, mat_var)
		else:
			self.nuclei_list = []
		

	def __read_nuclei(self,mat_path, mat_var):

		"Read the nuclei from .mat file."
			
		nuclei_coords = sp.io.loadmat(mat_path)[mat_var]
		
		#Nucleus = namedtuple("Nucleus", "x y z t confidence index")
		self.nuclei_list = []
	
		for i in xrange(nuclei_coords.shape[1]):
			x = int(nuclei_coords[0,i])
			y = int(nuclei_coords[1,i])
			z = int(nuclei_coords[2,i])
			self.nuclei_list.append(Nucleus(x,y,z, i))
	

		self.union_find = union_find.UnionFind(len(self.nuclei_list))
		
		self.nucleus_index_to_list_pos = dict((list_pos, nucleus.index) for list_pos, nucleus in enumerate(self.nuclei_list))
		
	def get_head_nucleus_in_its_set(self, nucleus):

		"Get the head nucleus of the union that this particular nucleu belongs to."

		nucleus_list_pos = self.nucleus_index_to_list_pos[nucleus.index]
		head_nucleus_list_pos = self.union_find.find(nucleus_list_pos)

		head_nucl = None
		if not self.union_find.is_deleted[head_nucleus_list_pos]:
			head_nucl = self.nuclei_list[head_nucleus_list_pos]

		return head_nucl



	def add_nucleus (self, nucleus):
	
		"Add a nucleus to the collection."
		self.nuclei_list.append(nucleus)
		self.nucleus_index_to_list_pos = dict((list_pos, nucleus.index) for list_pos, nucleus in enumerate(self.nuclei_list))
		self.union_find.add_one_item_to_union()


	def find_what_segment_each_nucleus_is_in(self, label_map):

		"What segment is associated with this nucleus."
		list_of_segment_labels = []

		for nucleus in self.nuclei_list:

			list_of_segment_labels.append (label_map[nucleus.x, nucleus.y, nucleus.z])

		return list_of_segment_labels


	def list_head_nuclei(self,):

		heads = set()

		for i in xrange(len(self.nuclei_list)):

			parent  = self.union_find.find(i)
			if not self.union_find.is_deleted[parent]:

				heads.add(self.nuclei_list[parent])

		return list(heads)
	

	def list_all_nuclei(self,):

		heads = set()

		for i in xrange(len(self.nuclei_list)):

			parent  = self.union_find.find(i)
			if not self.union_find.is_deleted[parent]:

				heads.add(self.nuclei_list[i])

		return list(heads)


		
	def find_closest_nucleus_to_segment(self, segment):
	

		"Given a segment, which one of the nuclei from this collection is associated with this segment."
		#x,y,z = zip(*segment.list_of_voxel_tuples)


		# pick only nuclei within this segment:	
		bbx = segment.bounding_box
		target_nucleus = None
		for nucleus in self.list_head_nuclei():
			
			try:
				if segment.mask[nucleus.x - bbx.xmin, nucleus.y - bbx.ymin, nucleus.z - bbx.zmin]:
					target_nucleus = nucleus
					break
			except:
				pass

		if target_nucleus is None:
			print "NO NUCLEUS IN SEGMENT!! Wtf."
			pdb.set_trace()


		return target_nucleus


	def delete_set_of_nucleus(self, nucleus):

		nucleus_list_pos = self.nucleus_index_to_list_pos[nucleus.index]
		self.union_find.delete_set_of(nucleus_list_pos)

	def delete_set_of_nucleus_by_index(self, nucleus_index):


		nucleus_list_pos = self.nucleus_index_to_list_pos[nucleus_index]
		self.union_find.delete_set_of(nucleus_list_pos)
		
		

	def merge_two_nuclei (self, nucleus1, nucleus2):
	
		"Merge two nuclei: union of nuclei."

		nucleus_list_pos1 = self.nucleus_index_to_list_pos[nucleus1.index]
		nucleus_list_pos2 = self.nucleus_index_to_list_pos[nucleus2.index]

		
		self.union_find.union(nucleus_list_pos1, nucleus_list_pos2)
		

