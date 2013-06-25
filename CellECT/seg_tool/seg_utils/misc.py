# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import sys

"""
Misc utilities:
	- print progress percentage (in place)
	- meshgrid for 3d made faster than numpy's
	- nonzero for 3d made faster than numpy's
"""

def print_progress(counter, total):

	"Print progress percentage in the same line in the terminal."

	progress = counter/float(total) *100
	try:
		sys.stdout.write("....... Progress: %f%%   \r" % (progress) )
		sys.stdout.flush()	
	except:
		import sys
		sys.stdout.write("....... Progress: %f%%   \r" % (progress) )
		sys.stdout.flush()	



def dld_meshgrid_3d(arr_x, arr_y, arr_z):

	"Meshgrid for 3d. Faster runtime for 3-D than numpy."

	X = Y = Z = np.zeros((len(arr_x) , len(arr_y), len(arr_z)))

	x_slice, y_slice = np.meshgrid(arr_x, arr_y)	

	for z in xrange(len(arr_z)):
		X[:,:,z] = x_slice
		Y[:,:,z] = z_slice
		Z[:,:,z] = np.ones(x_slice.shape) * arr_z[z]

	return X,Y,Z



def dld_nonzero3d(mat):

	""" 
	Nonzero for 3-d matrices.
	Faster than numpy for very sparse matrix where the nonzeros are in a bunch, such as supervoxel mask.
	"""

	xcoords = []
	ycoords = []
	zcoords = []


	for z in xrange(mat.shape[2]):

		if mat[:,:,z].sum() >0:
			(xs, ys) = np.nonzero(mat[:,:,z])
			zcoords.extend(np.ones(len(xs))*z)
			xcoords.extend(xs)
			ycoords.extend(ys)

	return (np.array(xcoords), np.array(ycoords), np.array(zcoords))
