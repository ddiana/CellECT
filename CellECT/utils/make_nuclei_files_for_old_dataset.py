import numpy as np
from scipy import misc
from scipy import io

nuclei = np.zeros((501,925,26))


for t in xrange(2):
	for i in xrange(26):

		nuclei[:,:,i] = misc.imread("/input_slices/%d.jpg" % (52*t + (i+1)*2))

		io.savemat("init_watershed_all_time_stamps/vol_nuclei_t_%d.mat" % t)

