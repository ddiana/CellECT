# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from pygraph.algorithms.accessibility import connected_components
import numpy as np
import scipy as sp
import pdb
import pylab

# Imports from this project
import CellECT
import CellECT.track_tool.globals


"""
UI class to visualize tracking results.
"""



class CellTrackerUI:

	"""
	UI for cell tracker object.
	Shows menu to choose ways to visualize the trakcing results:
	1) 3D point cloud of nuclei color coded by timestamp
	2) 3D point cloud of nuclei color coded by tracklet
	3) Histograms of cell sizes at each time stamp
	4) Track projections at one slice
	5) Nuclei at slice, color coded by track.
	6) Cell lineage visualization
	7) Segmentation color coded by tracklet
	"""

	

	def __init__(self, cell_tracker):

		self.cell_tracker =  cell_tracker
		self.thumbnail_dir = CellECT.__path__[0] + "/track_tool/resources/gui_thumbnails/"



	def plot_nuclei_at_timestamp_with_tracklet_coloring(self):
		"""
		3-d plot of nuclei (point cloud) color coded based on tracklet assignment
		"""
		
		cc_dict = connected_components(self.cell_tracker.graph)

		from mpl_toolkits.mplot3d import Axes3D

		fig = pylab.figure(figsize=(10,3))
		fig.canvas.set_window_title("3-D plot of nuclei color coded by tracklet")
	
		ax = fig.add_subplot(111, projection='3d')

		color_idx = np.linspace(0, 1, len(np.unique(cc_dict.values()))+1)
		np.random.shuffle(color_idx)


		x_vals = []
		y_vals = []
		z_vals = []
		colors = []


		for node in self.cell_tracker.graph.nodes():
			t,cp = self.cell_tracker.get_cell_profile_info_from_node_name(node)
			x_vals.append( self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles[cp].nucleus.x)
			y_vals.append( self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles[cp].nucleus.y)
			z_vals.append( self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles[cp].nucleus.z)
			colors.append(pylab.cm.jet( color_idx[cc_dict[node]])) 

	
		ax.scatter(x_vals, y_vals, z_vals, s=20, c = colors)
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')
		ax.pbaspect = [1., 1., 0.3]

		pylab.show()


	
	def plot_size_histograms_at_timestamp(self):
		"""
		plot of histogram of cell sizes, color coded from green to blue based on timestamp
		"""

		fig = pylab.figure(figsize=(10,7))
		fig.canvas.set_window_title("Cell size histograms color coded by timestamp.")

		color_idx = np.linspace(0, 1, len(self.cell_tracker.list_of_cell_profiles_per_timestamp))
	

		# histogram of cell sizes
		for colorshift, i in zip(color_idx, xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp))):
			bins = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].size_hist_bins[1:]
			hist = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].size_hist_vals
			ts = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].time_stamp
			pylab.plot(bins, hist , linewidth=3.0,color=pylab.cm.winter(colorshift), label="t = "+str(ts))
			pylab.hold(True)
		
		pylab.xlabel("Cell size histogram bins")
		pylab.ylabel("Normalized distribution")
		fig.canvas.set_window_title("Histogram of cell sizes for each time stamp")
	
		pylab.legend()
		pylab.show()



	def plot_nuclei_at_timestamp(self):
		"""
		3-d plot with all the nuclei color coded from green to blue based on time stamp
		"""

		from mpl_toolkits.mplot3d import Axes3D

		fig = pylab.figure(figsize=(10,3))
		fig.canvas.set_window_title("3-D plot of nuclei color coded by timestamp")
	
		ax = fig.add_subplot(111, projection='3d')

		color_idx = np.linspace(0, 1, len(self.cell_tracker.list_of_cell_profiles_per_timestamp))
	

		x_vals = []
		y_vals = []
		z_vals = []
		colors = []

		# histogram of cell sizes
		for colorshift, i in zip(color_idx, xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp))):
			x_vals.extend( [ cell_profile.nucleus.x for cell_profile in self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles] )
			y_vals.extend( [ cell_profile.nucleus.y for cell_profile in self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles])
			z_vals.extend( [ cell_profile.nucleus.z for cell_profile in self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles])
			colors.extend( [ pylab.cm.winter(colorshift) for i in xrange (len(self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles))])

		ax.scatter(x_vals, y_vals, z_vals, s=20, c = colors)
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')
		ax.pbaspect = [1., 1., 0.3]

		pylab.show()





	def plot_tracklets_one_slice(self, init_t, init_z):	
		"""
		draw one static slice with the nuclei and the tracklet trace color coded 
		with gradient based on time
		"""

		from matplotlib.widgets import Slider

		color_idx = np.linspace(0, 1, len(self.cell_tracker.list_of_cell_profiles_per_timestamp))

		def plot_tracklet_recursively(node, ax):

			t1, c1 = self.cell_tracker.get_cell_profile_info_from_node_name(node)
			cp1 = self.cell_tracker.list_of_cell_profiles_per_timestamp[t1].list_of_cell_profiles[c1]

			for inc_node in self.cell_tracker.graph.node_neighbors[node]:
				t2, c2 = self.cell_tracker.get_cell_profile_info_from_node_name(inc_node)
				cp2 = self.cell_tracker.list_of_cell_profiles_per_timestamp[t2].list_of_cell_profiles[c2]
				
				x = cp1.nucleus.x
				y = cp1.nucleus.y
				x2 = cp2.nucleus.x
				y2 = cp2.nucleus.y
		
				ax.plot([y,y2], [x, x2], color =  pylab.cm.winter(color_idx[t1]), linewidth = 2 )
				plot_tracklet_recursively(inc_node, ax)
		

			
			
		def plot_nuclei_with_tracklets(tval, zval, ax):

			depth_range = 5 # plot cells with nuclei in +/- depth_range

	
			for i in  xrange(tval,len(self.cell_tracker. list_of_cell_profiles_per_timestamp[tval].list_of_cell_profiles)):

				cp = self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].list_of_cell_profiles[i]
				if np.abs(cp.nucleus.z - zval) < depth_range:
					ax.plot( cp.nucleus.y, cp.nucleus.x, "ow" )		
					node = "t%d_c%d" % (tval, i)
					plot_tracklet_recursively(node, ax)		


		I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(init_t,init_z))+".tif")


		f = pylab.figure(figsize = (10,10))
		f.canvas.set_window_title("Tracklet projections at one slice")
		l1 = pylab.imshow(I, cmap="gray")
		ax1 = pylab.subplot(111)
		pylab.subplots_adjust(bottom=0.25)
		pylab.axis([0, I.shape[1], 0, I.shape[0]])
		plot_nuclei_with_tracklets(init_t, init_z,ax1 )


		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)
		s_t = Slider(ax_t, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_t)


		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z = pylab.axes([0.2, 0.05, 0.65, 0.03], axisbg=axcolor)
		s_z = Slider(ax_z, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z)
		
		t = init_t
		z = init_z
		z_old = [-1]
		t_old = [-1]


		# call back for time slider
		def update_t(val):

			t = int(s_t.val)
			z = int(s_z.val / 2) *2

			if (z_old[0] != z) or (t_old[0] !=t):

				I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")
				l1.set_data(I)
		
				# to remove old dots from current view
				ax1.lines = []

				plot_nuclei_with_tracklets(t,z, ax1)
				pylab.draw()
				z_old[0] = z
				t_old[0] = t


		# call back for z slider
		def update_z(val):

			t = int(s_t.val)
			z = int(s_z.val / 2) *2

			if (z_old[0] != z) or (t_old[0] !=t):
				I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")
				l1.set_data(I)

				ax1.lines = []
				# to remove old dots from current view
					
				plot_nuclei_with_tracklets(t,z, ax1)
				pylab.draw()
				z_old[0] = z
				t_old[0] = t
		
		
		s_t.on_changed(update_t)
		s_z.on_changed(update_z)

		pylab.show()




	def get_membrane_file_number_in_tif_sequence(self,t,z):
		return t* int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])*2 + z*2 +1




	def plot_color_tracklets_time_sequence(self, init_time, init_z):
		"""
		Plot a slice of the volume at given t and z, and draw the nuclei color coded
		according to tracklet slider bar to navigate time and z
		"""

		from matplotlib.widgets import Slider

		fig = pylab.figure(figsize =(10,10))
		fig.canvas.set_window_title("Plot tracklets as color coded nuclei at slice")


		depth_range = 5

		t = init_time
		z = init_z

		def draw_points_at_t_z(t,z, ax):
			# place color coded dots
			for i in  xrange(len(self. cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles)):

				cp = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles[i]
				if np.abs(cp.nucleus.z - z) < depth_range:
							
					node = "t%d_c%d" % (t, i)
					tracklet = cc_dict[node]
					color = pylab.cm.jet(color_idx[tracklet])
					ax.plot( cp.nucleus.y, cp.nucleus.x, "o", color = color)
					ax.text( cp.nucleus.y -0.2, cp.nucleus.x +0.2, tracklet, fontsize = 10, color = color )



		cc_dict = connected_components(self.cell_tracker.graph)
		color_idx = np.linspace(0, 1, len(np.unique(cc_dict.values()))+1)
		np.random.shuffle(color_idx)

		I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")


		#draw_points_at_t_z(init_time, init_z)

		# image subplot
		ax1 = pylab.subplot(111)
		pylab.subplots_adjust(bottom=0.25)
		l1 =  pylab.imshow(I, cmap = "gray")
		pylab.axis([0, I.shape[1], 0, I.shape[0]])
		draw_points_at_t_z(t,z,ax1)


		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)
		s_t = Slider(ax_t, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_time)


		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z = pylab.axes([0.2, 0.05, 0.65, 0.03], axisbg=axcolor)
		s_z = Slider(ax_z, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z)


		t = init_time
		z = init_z
		z_old = [-1]
		t_old = [-1]

		# call back for time slider
		def update_t(val):

			t = int(s_t.val)
			z = int(s_z.val / 2) *2

			if (z_old[0] != z) or (t_old[0] !=t):
				I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")
				l1.set_data(I)
		
				# to remove old dots from current view
				ax1.lines = []
				ax1.texts = []
					
				draw_points_at_t_z(t,z, ax1)
				pylab.draw()
				z_old[0] = z
				t_old[0] = t


		# call back for z slider
		def update_z(val):

			t = int(s_t.val)
			z = int(s_z.val / 2) *2

			if (z_old[0] != z) or (t_old[0] !=t):
				I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")
				l1.set_data(I)

				# to remove old dots from current view
				ax1.lines = []
				ax1.texts = []
					
				draw_points_at_t_z(t,z, ax1)
				pylab.draw()
				z_old[0] = z
				t_old[0] = t


		s_t.on_changed(update_t)
		s_z.on_changed(update_z)


		pylab.show()	



	def cell_lineage_gui(self, init_t1, init_t2, init_z1, init_z2):
		"""
		Display original slice and segmentation for two time stamps.
		When clicking on one cell, show the daughter cells in the second time stamp.
		Slider bars to navigate time and z.
		"""

		fig = pylab.figure(figsize=(35,11))
		fig.canvas.set_window_title("Cell lineage visualization")

		from matplotlib.widgets import Slider


		t1 = init_t1
		z1 = init_z1

		t2 = init_t2
		z2 = init_z2

		cc_of_interest = [-1]
		
		cc_dict = connected_components(self.cell_tracker.graph)
		
		I1 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t1,z1))+".tif")
		Seg1 = []
		Seg1.append(sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t1)+"_z_"+ str(z1) + "_seg.png"))
		
		Seg2 = []

	
		def make_cell_lineage_mask(tval, target_cc_index):

			labels = np.unique(Seg2[0])
			
			# make the segment border max value 
			max_val = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].list_of_cell_profiles)
			half_val = int(max_val / 2.)
			mask = np.uint(Seg2[0] == 0) * max_val

			for label in labels:
				if label>1:
					cp_index = self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].seg_label_to_cp_list_index[label]
					node_name = "t%d_c%d" % (tval, cp_index) 
					cc_index = cc_dict[node_name]
					
					if cc_index == target_cc_index:
						mask += np.uint(Seg2[0]==label) * half_val
						print "--- LINK: Label:", label,"| Tracklet:", cc_index,"| CellProfile:", cp_index, "| Node:", node_name, "| t:",tval
			return mask



		blank_page = [np.zeros(I1.shape)]		
		

		if init_t1 <= init_t2:
			I2 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t2,z2))+".tif")
			Seg2.append(sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t2)+"_z_"+ str(z2) + "_seg.png")	)
			lineage_mask = make_cell_lineage_mask(init_t2, cc_of_interest[0])
		else:
			I2 = blank_page[0]
			Seg2.append(blank_page[0])
			lineage_mask = blank_page[0]

		
		
		#draw_points_at_t_z(init_time, init_z)


		# image subplot
		ax1 = pylab.subplot(141)
		pylab.subplots_adjust(bottom=0.25)
		l1 =  pylab.imshow(I1, cmap = "gray")
		pylab.axis([0, I1.shape[1], 0, I1.shape[0]])

		# segmentation subplot
		ax2 = pylab.subplot(142)
		pylab.subplots_adjust(bottom=0.25)
		l2 =  pylab.imshow(Seg1[0], cmap = "spectral", vmin = 0, vmax = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[t1].list_of_cell_profiles), picker = True)
		pylab.axis([0, Seg1[0].shape[1], 0, Seg1[0].shape[0]])

		# image subplot
		ax3 = pylab.subplot(143)
		pylab.subplots_adjust(bottom=0.25)
		l3 =  pylab.imshow(I2, cmap = "gray")
		pylab.axis([0, I2.shape[1], 0, I2.shape[0]])

		# segmentation subplot
		ax4 = pylab.subplot(144)
		pylab.subplots_adjust(bottom=0.25)
		l4 =  pylab.imshow(lineage_mask, cmap = "gist_heat", vmin = 0, vmax = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[t2].list_of_cell_profiles))
		pylab.axis([0, Seg2[0].shape[1], 0, Seg2[0].shape[0]])



		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t1 = pylab.axes([0.2, 0.15, 0.25, 0.03], axisbg=axcolor)
		s_t1 = Slider(ax_t1, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_t1)


		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z1 = pylab.axes([0.2, 0.10, 0.25, 0.03], axisbg=axcolor)
		s_z1 = Slider(ax_z1, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z1)


		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t2 = pylab.axes([0.6, 0.15, 0.25, 0.03], axisbg=axcolor)
		s_t2 = Slider(ax_t2, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_t2)


		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z2 = pylab.axes([0.6, 0.10, 0.25, 0.03], axisbg=axcolor)
		s_z2 = Slider(ax_z2, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z2)


		t1 = init_t1
		z1 = init_z1
		z_old1 = [-1]
		t_old1 = [-1]

		t2 = init_t2
		z2 = init_z2
		z_old2 = [-1]
		t_old2 = [-1]


		I1_is_blank = [False]
		I2_is_blank = [t1 > t2]
		


		# call back for time slider
		def update_t1(val):

			t1 = int(s_t1.val)
			t2 = int(s_t2.val)
			z1 = int(s_z1.val / 2) *2
			z_seg1 = int(s_z1.val)

			if (z_old1[0] != z1) or (t_old1[0] !=t1):

				if t1<=t2:
					I1 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t1,z1))+".tif")
					Seg1[0] = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t1)+"_z_"+ str(z1) + "_seg.png")

					I1_is_blank[0] = False

					# if the other panel is set to blank_page, restore it..
					if I2_is_blank[0]:
						I2 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t2,z2))+".tif")
						Seg2[0] = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t2)+"_z_"+ str(z2) + "_seg.png")
						lineage_mask = make_cell_lineage_mask(t2, cc_of_interest[0])
						l3.set_data(I1)
						l4.set_data(lineage_mask)
						I2_is_blank[0] = False

				else:
					I1 = blank_page[0]
					Seg1[0] = blank_page[0]
					I1_is_blank[0] = True
				
				l1.set_data(I1)
				l2.set_data(Seg1[0])

				pylab.draw()

				z_old1[0] = z1
				t_old1[0] = t1



		# call back for z slider
		def update_z1(val):

			t1 = int(s_t1.val)
			t2 = int(s_t2.val)
			z1 = int(s_z1.val / 2) *2
			z_seg1 = int(s_z1.val)

			if (z_old1[0] != z1) or (t_old1[0] !=t1):
				if t1 <= t2:
					I1 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t1,z1))+".tif")
					Seg1[0] = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t1)+"_z_"+ str(z1) + "_seg.png")

					I1_is_blank[0] = False

					# if the other panel is set to blank_page, restore it..
					if I2_is_blank[0]:
						I2 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t2,z2))+".tif")
						Seg2[0] = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t2)+"_z_"+ str(z2) + "_seg.png")
						lineage_mask = make_cell_lineage_mask(t2, cc_of_interest[0])
						l3.set_data(I1)
						l4.set_data(lineage_mask)
						I2_is_blank[0] = False

				else:
					I1 = blank_page[0]
					Seg1[0] = blank_page[0]
					I1_is_blank[0] = True

				l1.set_data(I1)	
				l2.set_data(Seg1[0])

				pylab.draw()
	
				z_old1[0] = z1
				t_old1[0] = t1



		# call back for time slider
		def update_t2(val):

			t2 = int(s_t2.val)
			t1 = int(s_t1.val)
			z2 = int(s_z2.val / 2) *2
			z_seg2 = int(s_z2.val)

			if (z_old2[0] != z2) or (t_old2[0] !=t2):
				if t1 <= t2:
					I2 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t2,z2))+".tif")
					Seg2[0] = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t2)+"_z_"+ str(z2) + "_seg.png")

					I2_is_blank[0] = False

					# if the other panel is set to blank_page, restore it..
					if I1_is_blank[0]:
						I1 = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t1,z1))+".tif")
						Seg1[0] = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t1)+"_z_"+ str(z1) + "_seg.png")
						l1.set_data(I1)
						l2.set_data(Seg1[0])	
						I1_is_blank[0] = False

					lineage_mask = make_cell_lineage_mask(t2, cc_of_interest[0])
				else:
					I2 = blank_page[0]
					Seg2[0] = blank_page[0]
					lineage_mask = blank_page[0]
					I2_is_blank[0] = True

				l3.set_data(I2)
				l4.set_data(lineage_mask)

				pylab.draw()

				z_old2[0] = z2
				t_old2[0] = t2



		# call back for z slider
		def update_z2(val):

			t2 = int(s_t2.val)
			t1 = int(s_t1.val)
			z2 = int(s_z2.val / 2) *2
			z_seg2 = int(s_z2.val)

			if (z_old2[0] != z2) or (t_old2[0] !=t2):
				if t1 <= t2:
					I2 = sp.misc.imread(PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t2,z2))+".tif")
					Seg2[0] = sp.misc.imread(PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t2)+"_z_"+ str(z2) + "_seg.png")
					I2_is_blank[0] = False

					# if the other panel is set to blank_page, restore it..
					if I1_is_blank[0]:
						I1 = sp.misc.imread(PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t1,z1))+".tif")
						Seg1[0] = sp.misc.imread(PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t1)+"_z_"+ str(z1) + "_seg.png")
						l1.set_data(I1)
						l2.set_data(Seg1[0])
						I1_is_blank[0] = False

					lineage_mask = make_cell_lineage_mask(t2, cc_of_interest[0])
				else:
					I2 = blank_page[0]
					Seg2[0] = blank_page[0]
					lineage_mask = blank_page[0]
					I2_is_blank[0] = True

				l3.set_data(I2)
				l4.set_data(lineage_mask)

				pylab.draw()
	
				z_old2[0] = z2
				t_old2[0] = t2



		def onpick(event):

			xval = event.mouseevent.xdata
			yval = event.mouseevent.ydata
			tval = int(s_t1.val)
			label = Seg1[0][yval, xval]

			if label == 0:
				print "Border selected."
			if label == 1:
				print "Background selected."

			if label >1:
				cp_index = self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].seg_label_to_cp_list_index[label]

				node_name = "t" + str(tval) + "_c" + str(cp_index)
				cc_of_interest[0] = cc_dict[node_name]

				print "SELECTED: Label:", label, "| Tracklet:", cc_of_interest[0], "| CellProfile:", cp_index, "| Node:", node_name, "| t:", tval

				t2 = int(s_t2.val)
				lineage_mask = make_cell_lineage_mask( t2, cc_of_interest[0])

				l4.set_data(lineage_mask )
				pylab.draw()

	

		fig.canvas.mpl_connect('pick_event', onpick)
		s_t1.on_changed(update_t1)
		s_z1.on_changed(update_z1)
		s_t2.on_changed(update_t2)
		s_z2.on_changed(update_z2)

		pylab.show()



	def plot_tracklets_in_slice_with_seg(self, init_time, init_z):
		"""
		Plot a slice of the volume at given t and z, and draw the nuclei color coded
		according to tracklet plot the segmentation next to it. Color code segments
		and nuclei. Slider bar to navigate time and z.
		"""

		from matplotlib.widgets import Slider

		fig = pylab.figure(figsize=(18,10))
		fig.canvas.set_window_title("Segmentation color coded by tracklets")

		depth_range = 5

		t = init_time
		z = init_z

		cc_dict = connected_components(self.cell_tracker.graph)
		color_idx = np.linspace(0, 1, len(np.unique(cc_dict.values()))+1)
		np.random.shuffle(color_idx)


#		def make_colors_for_seg(Seg,tval):

#			my_colors = [pylab.cm.jet(color_idx[i]) for i in xrange(len(color_idx))]

#			for label in np.unique(Seg):
#				if label > 1:
#					cp_index = self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].seg_label_to_cp_list_index[label]
#					node = "t" + str(tval) + "_c" + str(cp_index) 
#					tracklet =  cc_dict[node]
#					my_colors[label]  =pylab.cm.jet(color_idx[tracklet]) 
#			
#			return my_colors
		

		def draw_points_at_t_z(t,z, ax):
			# place color coded dots
			for i in  xrange(len(self. cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles)):

				cp = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles[i]
				if np.abs(cp.nucleus.z - z) < depth_range:
							
					node = "t%d_c%d" % (t, i)
					tracklet = cc_dict[node]
					color = pylab.cm.jet(color_idx[tracklet])
					ax.plot( cp.nucleus.y, cp.nucleus.x, "o", color = color)
					ax.text(cp.nucleus.y, cp.nucleus.x, tracklet, fontsize=10, color = color )

		#TODO: Reassign color map instead of segmentation

		def get_label_to_tracklet_dict(Seg,tval):
			
			label_to_tracklet_dict = {}
			for label in np.unique(Seg):
				if label > 1:
					cp_index = self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].seg_label_to_cp_list_index[label]
					node = "t%d_c%d" % (tval, cp_index) 
					label_to_tracklet_dict[label] =  cc_dict[node]

			return label_to_tracklet_dict



		def relabel_to_tracklet(Seg, tval):
			
			label_to_tracklet_dict = get_label_to_tracklet_dict(Seg, tval)					

			for i in xrange (Seg.shape[0]):
				for j in xrange (Seg.shape[1]):
					label = Seg[i,j]
					if label > 1:
						Seg[i,j] = label_to_tracklet_dict[label]


		I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")
		Seg = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t)+"_z_"+ str(z) + "_seg.png")
		#draw_points_at_t_z(init_time, init_z)
		import time
		time1 = time.time()
		relabel_to_tracklet(Seg, init_time)
		print time.time() - time1
		
#		my_colors = make_colors_for_seg(Seg, init_time)

		# image subplot
		ax1 = pylab.subplot(121)
		pylab.subplots_adjust(bottom=0.25)
		l1 =  pylab.imshow(I, cmap = "gray")
		pylab.axis([0, I.shape[1], 0, I.shape[0]])
		draw_points_at_t_z(t,z,ax1)

		# segmentation subplot
		ax2 = pylab.subplot(122)
		pylab.subplots_adjust(bottom=0.25)
		l2 =  pylab.imshow(Seg, vmin = 0, vmax = len(color_idx), cmap = 'jet')
		pylab.axis([0, Seg.shape[1], 0, Seg.shape[0]])



		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)
		s_t = Slider(ax_t, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_time)


		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z = pylab.axes([0.2, 0.05, 0.65, 0.03], axisbg=axcolor)
		s_z = Slider(ax_z, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z)


		t = init_time
		z = init_z
		z_old = [-1]
		t_old = [-1]

		# call back for time slider
		def update_t(val):

			t = int(s_t.val)
			z = int(s_z.val / 2) *2
			z_seg = int(s_z.val)

			if (z_old[0] != z) or (t_old[0] !=t):
				I = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")
				l1.set_data(I)
				Seg = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t)+"_z_"+ str(z) + "_seg.png")
				relabel_to_tracklet(Seg, t)
				l2.set_data(Seg)

		
				# to remove old dots from current view
				ax1.lines = []
				ax1.texts = []
					
				draw_points_at_t_z(t,z, ax1)
				pylab.draw()

				z_old[0] = z
				t_old[0] = t



		# call back for z slider
		def update_z(val):

			t = int(s_t.val)
			z = int(s_z.val / 2) *2
			z_seg = int(s_z.val)

			if (z_old[0] != z) or (t_old[0] !=t):
				I = sp.misc.imread(PARAMETER_DICT["tif-slices-path"]+ "/"+str(self.get_membrane_file_number_in_tif_sequence(t,z))+".tif")
				l1.set_data(I)
				#print "loading ", "seg_all_time_stamps/timestamp_"+str(t)+"_z_"+ str(z) + "_seg.png"
				Seg = sp.misc.imread(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] + "/timestamp_"+str(t)+"_z_"+ str(z) + "_seg.png")
				relabel_to_tracklet(Seg, t)

				l2.set_data(Seg)

				# to remove old dots from current view
				ax1.lines = []
				ax1.texts = []
					
				draw_points_at_t_z(t,z, ax1)
				pylab.draw()
	
				z_old[0] = z
				t_old[0] = t


		s_t.on_changed(update_t)
		s_z.on_changed(update_z)


		pylab.show()


	def gui_menu(self):
		"""
		Main menu displaying all UIs.
		"""


		from matplotlib.widgets import Button

		def call_plot_nuclei_at_timestamp(event):
			self.plot_nuclei_at_timestamp()

		def call_plot_nuclei_at_timestamp_with_tracklet_coloring(event):
			self.plot_nuclei_at_timestamp_with_tracklet_coloring()

		def call_plot_size_histograms_at_timestamp(event):
			self.plot_size_histograms_at_timestamp()

		def call_plot_tracklets_one_slice(event):
			self. plot_tracklets_one_slice(1,15)

		def call_plot_color_tracklets_time_sequence(event):
			self.plot_color_tracklets_time_sequence(1,2)

		def call_cell_lineage_gui(event):
			self.cell_lineage_gui(1,1, 20, 20)
	
		def call_plot_tracklets_in_slice_with_seg(event):
			self. plot_tracklets_in_slice_with_seg(1,10)


		def enter_debug(event):
			pdb.set_trace()
		


		fig = pylab.figure(figsize=(10,7))
		fig.canvas.set_window_title("Cell tracker visualization menu")

		ax_button1 = pylab.axes([0.075, 0.85, 0.6, 0.075])
		button1 = Button(ax_button1, "3-D plot of nuclei color coded by timestamp")
		button1.on_clicked(call_plot_nuclei_at_timestamp)

		ax1 = pylab.axes( [0.725, 0.85, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "01.png")
		pylab.imshow(im)
		ax1.axis("off")

		

		ax_button2 = pylab.axes([0.075, 0.75, 0.6, 0.075])
		button2 = Button(ax_button2, "3-d plot of nuclei color coded by tracklet")
		button2.on_clicked(call_plot_nuclei_at_timestamp_with_tracklet_coloring)

		ax2 = pylab.axes( [0.725, 0.75, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "02.png")
		pylab.imshow(im)
		ax2.axis("off")




		ax_button3 = pylab.axes([0.075, 0.65, 0.6, 0.075])
		button3 = Button(ax_button3, "Histograms of cell sizes at each timestamp")
		button3.on_clicked(call_plot_size_histograms_at_timestamp)

		ax3 = pylab.axes( [0.725, 0.65, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "03.png")
		pylab.imshow(im)
		ax3.axis("off")



		ax_button4 = pylab.axes([0.075, 0.55, 0.6, 0.075])
		button4 = Button(ax_button4, "Draw track projections at one slice")
		button4.on_clicked(call_plot_tracklets_one_slice)

		ax4 = pylab.axes( [0.725, 0.55, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "04.png")
		pylab.imshow(im)
		ax4.axis("off")




		ax_button5 = pylab.axes([0.075, 0.45, 0.6, 0.075])
		button5 = Button(ax_button5, "Plot tracklets as color coded nuclei at slice")
		button5.on_clicked(call_plot_color_tracklets_time_sequence)
	
		ax5 = pylab.axes( [0.725, 0.45, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "05.png")
		pylab.imshow(im)
		ax5.axis("off")





		ax_button6 = pylab.axes([0.075, 0.35, 0.6, 0.075])
		button6 = Button(ax_button6, "Cell lineage visualization")
		button6.on_clicked(call_cell_lineage_gui)

		ax6 = pylab.axes( [0.725, 0.35, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "06.png")
		pylab.imshow(im)
		ax6.axis("off")




		ax_button7 = pylab.axes([0.075, 0.25, 0.6, 0.075])
		button7 = Button(ax_button7, "Segmentation color coded by tracklets")
		button7.on_clicked(call_plot_tracklets_in_slice_with_seg)

		ax7 = pylab.axes( [0.725, 0.25, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "07.png")
		pylab.imshow(im)
		ax7.axis("off")



		ax_button8 = pylab.axes([0.075, 0.05, 0.6, 0.075])
		button8 = Button(ax_button8, "Enter debug...")
		button8.on_clicked(enter_debug)

		ax8 = pylab.axes( [0.725, 0.05, 0.2, 0.075])
		im = sp.misc.imread(self.thumbnail_dir+ "/" + "08.png")
		pylab.imshow(im)
		ax8.axis("off")

		pylab.show()
		
