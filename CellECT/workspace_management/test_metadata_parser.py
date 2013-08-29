# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import unittest
import pdb

# Imports from this project
from CellECT.workspace_management import metadata



class ExtractMetadataFromTestSequence(unittest.TestCase):


	def correct_sample(self, line, meta_field_string_to_exec, expected_meta_value):
		# check that the field from the metadata gets the expected_meta_value from the line.

		meta = metadata.Metadata()
		meta.get_meta_from_line(line)
		exec meta_field_string_to_exec
		self.assertEqual(meta_field, expected_meta_value)


	def incorrect_sample(self, line, meta_field_string_to_exec):
		# check that the value is not picked up

		meta = metadata.Metadata()
		meta.get_meta_from_line(line)
		exec meta_field_string_to_exec
		self.assertEqual(meta_field, None)


	def test_simple_equal(self):
		# check that "YResolution= 3" gives meta.yres == 3

		line = "YResolution= 3"
		self.correct_sample(line, "meta_field = meta.yres", 3)
		

	def test_simple_colon(self):
		# check that "YResolution: 3" gives meta.yres == 3

		line = "YResolution: 3"
		self.correct_sample(line, "meta_field = meta.yres", 3)
		

	def test_white_spaces_colon(self):
		# check that white spaces don't affect it'

		line = "        YResolution       \t : 3     "
		self.correct_sample(line, "meta_field = meta.yres", 3)


	def test_white_spaces_quotations_colon(self):
		# check that white spaces don't affect it'

		line = '        YResolution       \t : "3"     '
		self.correct_sample(line, "meta_field = meta.yres", 3)


	def test_white_spaces_quotations_colon(self):
		# check that white spaces don't affect it'

		line = '        YResolution       \t : "3"     '
		self.correct_sample(line, "meta_field = meta.yres", 3)


	def test_alphanumeric_values_not_picked_up(self):
		# check that YResolution = 203d3 is not picked up
		line = '        YResolution       \t : "203d3"     '
		self.incorrect_sample(line, "meta_field = meta.yres")


	def test_alpha_values_not_picked_up(self):
		# check that YResolution = fsdg is not picked up
		line = '        YResolution       \t : "gsrset"     '
		self.incorrect_sample(line, "meta_field = meta.yres")


	def test_wrong_info_is_not_picked_up(self):
		# check that YResolution-3 : 3 is not picked up
		line = '  YResolution-3 : 3     '
		self.incorrect_sample(line, "meta_field = meta.yres")


if __name__ == "__main__":

	unittest.main()
