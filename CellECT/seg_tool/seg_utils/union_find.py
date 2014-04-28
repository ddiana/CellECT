# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb

"""
Implementation of union find. 
This is used by the nuclei collection for merging segments.
"""

class UnionFind(object):

	def __init__ (self, number_lements):
		self.parents = range(number_lements)
		self.set_size = [1] * number_lements
		self.is_deleted = [False] * number_lements



	def add_one_item_to_union(self):

		"Add an item to the existing union."

		self.parents.append(len(self.parents))
		self.set_size.append(1)
		self.is_deleted.append(False)

	
	def find(self, element):

		"Find the head of the set that 'element' is in."

		while (self.parents[element] != element):
			element = self.parents[element]
		return element


	def delete_set_of(self, element):

		root = self.find(element)
		self.is_deleted[root] = True


	# TODO check if you need to call find for set size.	
	def union (self, element1, element2):

		"Union of two elements."

		"If one is deleted, then both are deleted."

		root_element1 = self.find(element1)
		root_element2 = self.find(element2)

		if root_element1 != root_element2:
#			if self.set_size[root_element1] < self.set_size[root_element2]:
#				self.parents[root_element1] = root_element2
#			elif self.set_size[root_element1] > self.set_size[root_element2]:
#				self.parents[root_element2] = root_element1
#			else:
#				self.parents[root_element2] = root_element1
#				self.set_size[root_element1] = self.set_size[root_element1] +1

			if self.set_size[root_element1] > self.set_size[root_element2]:
 				self.set_size[root_element1] += self.set_size[root_element2]
				self.parents[root_element2] = root_element1
			else:
				self.set_size[root_element2] += self.set_size[root_element1]
				self.parents[root_element1] = root_element2

			self.is_deleted[root_element1] = self.is_deleted[root_element1] or self.is_deleted[root_element2]
			self.is_deleted[root_element2] = self.is_deleted[root_element1] or self.is_deleted[root_element2]
