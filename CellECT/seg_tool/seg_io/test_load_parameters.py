# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import unittest

# Imports from this project
from CellECT.seg_tool.seg_io.load_parameters import parse_config_file_line as parser;

import CellECT.seg_tool.globals



class ParseConfigFileTestSequence(unittest.TestCase):

	def correct_sample(self, line, expected_key, expected_val):

		required_keys = CellECT.seg_tool.globals.default_parameter_dictionary_keys

		all_keys = set(CellECT.seg_tool.globals.default_parameter_dictionary_keys)
		all_keys = all_keys.union(set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_bq_only))
		all_keys = all_keys.union(set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_cellness_metric_only))


		key, val = parser(line, required_keys, all_keys)
		self.assertEqual(key , expected_key)
		self.assertEqual(val, expected_val)	

	def fail_sample(self, line):
		
		required_keys = CellECT.seg_tool.globals.default_parameter_dictionary_keys

		all_keys = set(CellECT.seg_tool.globals.default_parameter_dictionary_keys)
		all_keys = all_keys.union(set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_bq_only))
		all_keys = all_keys.union(set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_cellness_metric_only))



		self.assertRaises(IOError, parser, line, required_keys, all_keys)

	
	def test_simple_sample(self):
		# simple key = value
		self.correct_sample("training_vol_nuclei_mat_var=seeds", "training_vol_nuclei_mat_var", "seeds")


	def test_with_white_spaces(self):
		# check that white spaces don't affect it
		self.correct_sample("		  training_vol_nuclei_mat_var       = seeds	", "training_vol_nuclei_mat_var", "seeds")


	def test_empty_line(self):
		# check that empty lines get skipped
		self.correct_sample("", None, None)


	def test_exception_for_missing_value(self):
		# check that it raises exception for missing value in key=value pattern
		self.fail_sample("training_vol_nuclei_mat_var = ")

	
	def test_exception_for_missing_key(self):
		# check that it raises exception for missing key in key=value pattern
		self.fail_sample(" = seeds")


	def test_exception_for_missing_key_value(self):
		# check that it fails for missing key and value in key=value pattern
		self.fail_sample("=")

	def test_exception_for_missing_pattern(self):
		# check that it raises exception for missing key=value pattern
		self.fail_sample("training_vol_nuclei_mat_var seeds")


	def test_exception_for_bad_key(self):
		# check that it raises exception for a key that is not in the dictionary	
		self.fail_sample("training_vol_nuclei_mat_va = seeds")


	def test_exception_for_bad_pattern_two_equals(self):
		# check that it raises exception for a bad key=value patter	
		self.fail_sample("training_vol_nuclei_mat_va = seeds = seeds")
	




if __name__ == "__main__":

	unittest.main()

