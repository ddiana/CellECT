# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import scipy as sp
import pdb
import pylab


"""
Shows all the seeds/nuclei that will be used as Watershed seeds.
The items added by the user are shown in blue.
The items from the nuclei detector are shown in red.
"""



def show_new_user_points(original_init_pts, init_pts):

	from mpl_toolkits.mplot3d import Axes3D
		
	fig = pylab.figure(figsize=(10,3))
	ax = fig.add_subplot( 111, projection='3d')
	
	x_vals = [ original_init_pts [0, i] for i in xrange (original_init_pts.shape[1])]
	y_vals = [ original_init_pts [1, i] for i in xrange (original_init_pts.shape[1])]
	z_vals = [ original_init_pts [2, i] for i in xrange (original_init_pts.shape[1])]
	colors = [ 'r' for i in xrange (original_init_pts.shape[1])]
	#markers = [ 'o' for i in xrange (original_init_pts.shape[1]) ]
	
	x_vals.extend( [ init_pts [0, i] for i in xrange (original_init_pts.shape[1], init_pts.shape[1])] )
	y_vals.extend( [ init_pts [1, i] for i in xrange (original_init_pts.shape[1], init_pts.shape[1])] )
	z_vals.extend( [ init_pts [2, i] for i in xrange (original_init_pts.shape[1], init_pts.shape[1])] )
	colors.extend( [ 'b' for i in xrange (original_init_pts.shape[1], init_pts.shape[1])] )
	#markers.extend( [ 'o' for i in xrange (original_init_pts.shape[1], init_pts.shape[1])]  )
	
	ax.scatter(x_vals, y_vals, z_vals, s=20, c = colors)
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	ax.pbaspect = [1., 1., 0.3]
	#ax.title("OLD SEEDS (red circle) and NEW SEEDS (blue triangle)")
	
	pylab.show()
	pylab.close()

