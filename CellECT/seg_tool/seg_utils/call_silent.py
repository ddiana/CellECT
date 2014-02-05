# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports

import pdb
import sys
import os

"""
Functions to silence the output of a function or process.
"""


def call_silent(f, *args, **kw_args):
	"Redirent stdout to null."

	old_stdout = sys.stdout
	sys.stdout = open('/dev/null', 'w')
	r1 = f(*args, **kw_args)
	sys.stdout.close()
	sys.stdout = old_stdout
	return r1

def call_silent_err(f, *args, **kw_args):
	"Redirect stderr to null."

	old_stderr = sys.stderr
	sys.stderr = open('/dev/null', 'w')
	r1 = f(*args, **kw_args)
	sys.stderr.close()
	sys.stderr = old_stderr
	return r1


def call_silent_process(f, *args, **kw_args):
	"Redirect stdout to null, for a newly spawned process."

	#devnull = open('/dev/null', 'w')
	#oldstdout_fno = os.dup(sys.stdout.fileno())
	#os.dup2(devnull.fileno(), 1)
	#r1 = f(*args, **kw_args)
	#os.dup2(oldstdout_fno, 1)
	#devnull.close()
	
#	so, sys.stdout = sys.stdout, StringIO()
	r1 = f(*args, **kw_args)
#	sys.stdout = so

	return r1

def call_silent_process_err(f, *args, **kw_args):
	"Redirect stdin to null, for a newly spawned process."

	devnull = open('/dev/null', 'w')
	oldstderr_fno = os.dup(sys.stderr.fileno())
	os.dup2(devnull.fileno(), 1)
	r1 = f(*args, **kw_args)
	os.dup2(oldstderr_fno, 1)
	devnull.close()
	return r1

def call_silent_popen(f, *args, **kw_args):

	old_stdout = sys.stdout
	sys.stdout = open('/dev/null', 'w')
	r1 = f(*args, **kw_args)
	sys.stdout.close()
	sys.stdout = old_stdout
	return r1


