# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb



class Seed(object):

	"Additional seed to an existing nucleus."
	
	def __init__ (self, x,y,z,nucleus_index, index):
	
		self.x = x
		self.y = y
		self.z = z
		self.nucleus_index = nucleus_index	
		self.index = index

class SeedCollection(object):

	"Collection of seeds."

	def __init__ (self, list_of_seeds):
	
		self.list_of_seeds = list_of_seeds
		
	def add_seed (self, new_seed):
	
		self.list_of_seeds.append(new_seed)
	

