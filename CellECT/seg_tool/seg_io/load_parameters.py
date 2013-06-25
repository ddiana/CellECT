# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb
import re

# Imports from this project
import globals


def read_program_parameters(config_file_path):

	"Read parameters from config file."

	f = open(config_file_path)
	line = f.readline()
	
	while (line != ""):
		matches = re.match("(.*)=(.*)", line)
		if matches:
			key = matches.group(1).strip()
			val = matches.group(2).strip()
			globals.DEFAULT_PARAMETER[key] = val
		else:
			print "ERROR reading config file"
		line = f.readline()

