# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from pygraph.algorithms.accessibility import connected_components
import numpy as np
import scipy as sp
import pdb
import pylab
import random
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import os.path

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
		self.thumbnail_dir = os.path.join(CellECT.__path__[0] , "track_tool","resources","gui_thumbnails","")

		colors = [(0,0,0)] + [(random.random(),random.random(),random.random()) for i in xrange(255)]
		self.color_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

	def plot_nuclei_at_timestamp_with_tracklet_coloring(self):
		"""
		3-d plot of nuclei (point cloud) color coded based on tracklet assignment
		"""
		
		cc_dict = connected_components(self.cell_tracker.graph)

		from mpl_toolkits.mplot3d import Axes3D

		fig = pylab.figure(figsize=(10,3), facecolor='white')
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



	def plot_feature_histograms_over_time(self, feat):

		fig = pylab.figure(figsize=(7,3), facecolor='white')
		fig.canvas.set_window_title("Hist over time: %s" % feat)

		color_idx = np.linspace(0, 1, len(self.cell_tracker.list_of_cell_profiles_per_timestamp))
	
		bins = self.cell_tracker.list_of_cell_profiles_per_timestamp[0].feature_histograms[feat][1]

		# histogram of cell sizes
		for colorshift, i in zip(color_idx, xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp))):

			hist = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].feature_histograms[feat][0]
			ts = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].time_stamp
			pylab.plot(bins[:-1], hist , linewidth=3.0,color=pylab.cm.jet(colorshift), label="t = "+str(ts))
			pylab.hold(True)
		
		pylab.xlabel("Histogram bins")
		pylab.ylabel("p.d.f.")
		pylab.grid()
		fig.canvas.set_window_title("Histogram over time: %s" % feat)
	


	def plot_2dhist_at_timestamp(self, feature_pair, t):

		feat1 = feature_pair[0]
		feat2 = feature_pair[1]

		fig = pylab.figure(figsize=(6,5), facecolor='white')
		fig.canvas.set_window_title("Hist %s Vs %s at t=%d" % (feat1, feat2, t))

		hist, bins1, bins2 = self. cell_tracker.list_of_cell_profiles_per_timestamp[t].feature_histograms[feature_pair]
		
		pylab.imshow(hist, extent = [min(bins1), max(bins1), min(bins2), max(bins2)], cmap="Blues",  aspect='auto')

		pylab.colorbar()
		pylab.xlabel(feat1)
		pylab.ylabel(feat2)
		pylab.grid()

	def plot_size_histograms_at_timestamp(self):
		"""
		plot of histogram of cell sizes, color coded from green to blue based on timestamp
		"""

		fig = pylab.figure(figsize=(7,5), facecolor='white')
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

	
	def plot_feature_scatter_plot(self, feat1, feat2, feat3, title_text, axis1_text, axis2_text, axis3_text):


		fig = pylab.figure(figsize=(7,5), facecolor='white')
		fig.canvas.set_window_title(title_text)

		color_idx = np.linspace(0, 1, len(self.cell_tracker.list_of_cell_profiles_per_timestamp))
		feat1_vals = []
		feat2_vals = []	
		feat3_vals = []
		colors = []
		
		ax = fig.add_subplot(111)

		if not feat3 is None:
			ax = fig.add_subplot(111, projection='3d')
		

		# histogram of cell sizes
		for colorshift, i in zip(color_idx, xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp))):
			feat1_vals.extend( [self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles[idx].dict_of_features[feat1] for idx in xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles))]	)
			feat2_vals.extend( [self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles[idx].dict_of_features[feat2] for idx in xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles))]	)
			if not feat3 is None:
				feat3_vals.extend( [self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles[idx].dict_of_features[feat3] for idx in xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles))]	)

			colors.extend( [ pylab.cm.winter(colorshift) for i in xrange (len(self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles))])

		if not feat3 is None:
			ax.scatter(feat1_vals, feat2_vals, feat3_vals, c = colors)
		else:
			ax.scatter(feat1_vals, feat2_vals, c = colors)

		pylab.hold(True)

		ax.set_xlabel(axis1_text)
		ax.set_ylabel(axis2_text)
		if not feat3 is None:
			ax.set_zlabel(axis3_text)

		fig.canvas.set_window_title(title_text)
	
		pylab.legend()


	def plot_dist_to_nucleus_hists(self):

		fig = pylab.figure(figsize=(7,5), facecolor='white')
		fig.canvas.set_window_title("Distance to nucleus histograms")

		num_bins = len(self.cell_tracker.list_of_cell_profiles_per_timestamp[0].list_of_cell_profiles[0].dict_of_features["border_to_nucleus_dist_hist"])
		color_idx = np.linspace(0, 1, num_bins )
	
		bins = np.arange(0,1, 1/float(num_bins))


		# histogram of cell sizes
		for colorshift, i in zip(color_idx, xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp))):
			ts = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].time_stamp
			hists = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles[0].dict_of_features["border_to_nucleus_dist_hist"]
			for idx in  xrange(1,len(self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles)):
				cp_item = self.cell_tracker.list_of_cell_profiles_per_timestamp[i].list_of_cell_profiles[idx]
				hist = cp_item.dict_of_features["border_to_nucleus_dist_hist"]
				hist = np.array(hist) / float(np.sum(hist))
				hists = np.vstack((hists, hist))

			hist = hists.mean(0)
			pylab.plot(bins, hist , linewidth=3.0, color=pylab.cm.winter(colorshift), label="t = "+str(ts))
			pylab.hold(True)
		
		pylab.xlabel("")
		pylab.ylabel("")
		fig.canvas.set_window_title("Distance to nucleus histograms")
	
		pylab.legend()



	def plot_groups_at_time_point(self,t, feat1, feat2):


		markers = ['ro', 'go', 'bo', 'yo', 'ko', 'mo', 'co']

		cp_list = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles



		fig = pylab.figure( facecolor='white')

		counter = -1

		for group_name in self.groups.keys():

			counter +=1
			gr = self.groups[group_name][t]

			feat1_vals = []
			feat2_vals = []
			
			for idx in gr:
				feat1_vals.append(cp_list[idx].dict_of_features[feat1])
				feat2_vals.append(cp_list[idx].dict_of_features[feat2])

			pylab.plot(feat1_vals, feat2_vals, markers[counter], label = group_name)

			pylab.hold(True)

		fig.canvas.set_window_title("Time point %s" % t)

		pylab.legend(loc="best")
		pylab.xlabel(feat1)
		pylab.ylabel(feat2)
		pylab.grid()



	def shape_hist_for_groups_at_time_point(self, t):


		markers = ['r-', 'g-', 'b-', 'y-', 'k-', 'm-', 'c-']


		num_time_points = len(self.cell_tracker.list_of_cell_profiles_per_timestamp)

		fig = pylab.figure( facecolor='white')

		counter = -1

		for group_name in self.groups.keys():

			bin_count = len(self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles[0].dict_of_features["border_to_nucleus_dist_hist"])
			feat_average = np.zeros((1,bin_count))

			counter +=1

			gr = self.groups[group_name][t]
			cp_list = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles
	
			for idx in gr:
				feat_average += cp_list[idx].dict_of_features["border_to_nucleus_dist_hist"]
		
			feat_average = feat_average /float(feat_average.sum())

	
			pylab.plot(range(bin_count), feat_average.T, markers[counter], label = group_name, linewidth = 2)

			pylab.hold(True)

		fig.canvas.set_window_title("Distance of border to nucleus at t=%d" % t)

		pylab.legend(loc="best")
		pylab.xlabel("Bin index")
		pylab.ylabel("p.d.f.")
		pylab.grid()



	def plot_group_average_per_time(self,feat):


		markers = ['ro-', 'go-', 'bo-', 'yo-', 'ko-', 'mo-', 'co-']



		num_time_points = len(self.cell_tracker.list_of_cell_profiles_per_timestamp)

		fig = pylab.figure( facecolor='white')

		counter = -1

		for group_name in self.groups.keys():


			feat_average = []

			counter +=1
	
			for t in xrange(num_time_points):

				gr = self.groups[group_name][t]
				cp_list = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles
	
				sum_val = 0
				for idx in gr:
					sum_val += cp_list[idx].dict_of_features[feat]

				feat_average.append(sum_val/float(len(self.groups[group_name][t])))
	
			pylab.plot(range(num_time_points), feat_average, markers[counter], label = group_name, linewidth = 2)

			pylab.hold(True)

		fig.canvas.set_window_title("%s over time" % feat)

		pylab.legend(loc="best")
		pylab.xlabel("Time point")
		pylab.ylabel(feat)
		pylab.grid()







	def tissue_groups(self):


		group_names = ["notochord_group", "muscle_group", "skin_group", "head_group"]
		num_time_points = len(self.cell_tracker.list_of_cell_profiles_per_timestamp)

		self.groups = {}
		for name in group_names:
			self.groups[name] = dict((x,set()) for x in xrange(num_time_points))
	
		for t in xrange(num_time_points):
			for c in xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles)):

				cp = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles[c]

				# notochord_group:
				if cp.dict_of_features["dist_to_AP_axis"] < 10 and cp.dict_of_features["position_along_AP_axis"] > 60:
					self.groups["notochord_group"][t].add(c)

				# muscle_group:
				if cp.dict_of_features["dist_to_AP_axis"] < 20 and cp.dict_of_features["dist_to_AP_axis"]>5 and cp.dict_of_features["position_along_AP_axis"] > 60 and cp.dict_of_features["centroid_dist_from_margin"]>10:
					self.groups["muscle_group"][t].add(c)

				# skin_group
				if cp.dict_of_features["centroid_dist_from_margin"] <15:
					self.groups["skin_group"][t].add(c)

				# head_group
				if cp.dict_of_features["position_along_AP_axis"] < 50 and cp.dict_of_features["centroid_dist_from_margin"] >5 and cp.dict_of_features["dist_to_AP_axis"] < 25:
					self.groups["head_group"][t].add(c)


	def plot_shape_hists_per_group(self):




		fig = pylab.figure( facecolor='white')
		fig.canvas.set_window_title("Cell shape histograms per tissue over time")

		color_idx = np.linspace(0, 1, len(self.cell_tracker.list_of_cell_profiles_per_timestamp))

		ax1 = fig.add_subplot(141)
		ax2 = fig.add_subplot(142)
		ax3 = fig.add_subplot(143)
		ax4 = fig.add_subplot(144)

		num_bins = len(self.cell_tracker.list_of_cell_profiles_per_timestamp[0].list_of_cell_profiles[0].dict_of_features["border_to_nucleus_dist_hist"])

		bins = np.arange(0,1, 1/float(num_bins))

		ax_list = [ax1, ax2, ax3, ax4]

		counter = -1

		for group_name in self.groups.keys():

			bin_count = len(self.cell_tracker.list_of_cell_profiles_per_timestamp[0].list_of_cell_profiles[0].dict_of_features["border_to_nucleus_dist_hist"])
			feat_average = np.zeros((1,bin_count))

			counter +=1

			ax = ax_list[counter]

			for colorshift, t in zip(color_idx, xrange(len(self.cell_tracker.list_of_cell_profiles_per_timestamp))):
				gr = self.groups[group_name][t]


				cp_list = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles
	
				for idx in gr:
					feat_average += cp_list[idx].dict_of_features["border_to_nucleus_dist_hist"]
		
					feat_average = feat_average /float( np.sum(feat_average))

	
				ax.plot(range(bin_count), feat_average.T,color=pylab.cm.winter(colorshift) , label = group_name, linewidth = 2)

			ax.set_title(group_name)
			ax.set_xlabel("Cell size histogram bins")
			ax.set_ylabel("Normalized distribution")
		fig.canvas.set_window_title("Histogram of cell sizes for each time stamp")



	




	def global_feature_plots(self,):



		markers = ['ro-', 'go-', 'bo-', 'yo-', 'ko-', 'mo-', 'co-']

		num_time_points = len(self.cell_tracker.list_of_cell_profiles_per_timestamp)

		fig = pylab.figure( facecolor='white')

		ax1 = fig.add_subplot(131)
		ax2 = fig.add_subplot(132)
		ax3 = fig.add_subplot(133)


		counter = -1



		vol_sum = []
		vol_avg = []
		num_segs = []

		counter +=1
	
		for t in xrange(num_time_points):
			cp_list = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles
	
			sum_val = 0
			for cp in cp_list:
				sum_val += cp.dict_of_features["volume_by_res"]

			vol_sum.append(sum_val)
			vol_avg.append(sum_val/float(len(cp_list)))
			num_segs.append(len(cp_list))
	
		title_list = ["Total volume over time", "Total cells over time","Avg cell vol over time"]		
		ax_list = [ax1, ax2, ax3]
		val_list = [vol_sum, num_segs, vol_avg]
		feat_list = ["cubic microns", "count", "cubic microns"]

#		pdb.set_trace()

		for i in xrange(len(ax_list)):
			ax_list[i].plot(range(num_time_points), val_list[i], markers[counter], linewidth = 2)
			ax_list[i].set_title( title_list[i])
			ax_list[i].hold(True)
				
			#ax_list[i].legend(loc="best")
			ax_list[i].set_xlabel("Time point")
			ax_list[i].set_ylabel(feat_list[i])
			ax_list[i].grid()

		fig.canvas.set_window_title("Global values over time" )




	def groups_in_space(self, t):

		markers = ['ro', 'go', 'bo', 'yo', 'ko', 'mo', 'co']

		cp_list = self.cell_tracker.list_of_cell_profiles_per_timestamp[t].list_of_cell_profiles



		fig = pylab.figure( facecolor='white')

		ax = fig.add_subplot(111, projection='3d')

		counter = -1

		for group_name in self.groups.keys():

			counter +=1
			gr = self.groups[group_name][t]

			x_vals = []
			y_vals = []
			z_vals = []
			
			for idx in gr:
				x_vals.append(cp_list[idx].dict_of_features["centroid_res"][0])
				y_vals.append(cp_list[idx].dict_of_features["centroid_res"][1])
				z_vals.append(cp_list[idx].dict_of_features["centroid_res"][2])


			pylab.plot(x_vals, y_vals, z_vals, markers[counter], label = group_name)

			pylab.hold(True)

		fig.canvas.set_window_title("Groups at time point %d " %t)

		pylab.legend(loc="best")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		pylab.grid()


	def plot_stats(self):


		last_time_point = len(self.cell_tracker.list_of_cell_profiles_per_timestamp) -1

		has_groups = True
		if has_groups:


	
		#self.plot_size_histograms_at_timestamp()

		#"dist_to_AP_axis", "angle_with_AP_axis", "position_along_AP_axis", "size",  "min_distance_from_margin" ]
		#self.plot_feature_scatter_plot("size", "dist_to_AP_axis", "position_along_AP_axis", "Size, Distance to AP axis, Position along AP axis", "size", "distance to AP", "percent along AP")
		#self.plot_feature_scatter_plot("size", "dist_to_AP_axis", None, "Size Vs Distance to AP axis", "size", "distance to AP", None)
		#self.plot_feature_scatter_plot("size", "position_along_AP_axis", None, "Size Vs Position along AP axis", "size", "percent along AP", None)
		#self.plot_feature_scatter_plot("size", "angle_with_AP_axis", None, "Size Vs Angle with AP axis", "size", "angle with AP", None)
		#self.plot_feature_scatter_plot("position_along_AP_axis", "dist_to_AP_axis", "angle_with_AP_axis", "Position relative to AP axis", "percent along axis", "dist to axis", "angle with axis")
		
#		self.plot_feature_scatter_plot("position_along_AP_axis", "sphericity", None, "Position relative to AP axis", "percent along axis",  "sphericity", None)
#		self.plot_feature_scatter_plot("dist_to_AP_axis", "sphericity", None, "Position relative to AP axis", "distance to axis",  "sphericity", None)
#		self.plot_feature_scatter_plot("centroid_dist_from_margin", "sphericity", None, "Position relative to AP axis", "distance to margin",  "sphericity", None)

#		self.plot_feature_scatter_plot("position_along_AP_axis", "cylindricity", None, "Position relative to AP axis", "percent along axis",  "cylindricity", None)
#		self.plot_feature_scatter_plot("dist_to_AP_axis", "cylindricity", None, "Position relative to AP axis", "distance to axis",  "cylindricity", None)
#		self.plot_feature_scatter_plot("centroid_dist_from_margin", "cylindricity", None, "Position relative to AP axis", "distance to margin",  "cylindricity", None)

#		self.plot_feature_scatter_plot("position_along_AP_axis", "volume_by_res", None, "Position relative to AP axis", "percent along axis",  "volume_by_res", None)
#		self.plot_feature_scatter_plot("dist_to_AP_axis", "volume_by_res", None, "Position relative to AP axis", "distance to axis",  "volume_by_res", None)
#		self.plot_feature_scatter_plot("centroid_dist_from_margin", "volume_by_res", None, "Position relative to AP axis", "distance to margin",  "volume_by_res", None)

#		self.plot_feature_scatter_plot("position_along_AP_axis", "flatness", None, "Position relative to AP axis", "percent along axis",  "flatness", None)
#		self.plot_feature_scatter_plot("dist_to_AP_axis", "flatness", None, "Position relative to AP axis", "distance to axis",  "flatness", None)
#		self.plot_feature_scatter_plot("centroid_dist_from_margin", "flatness", None, "Position relative to AP axis", "distance to margin",  "flatness", None)

#		self.plot_feature_scatter_plot("position_along_AP_axis", "elongation", None, "Position relative to AP axis", "percent along axis",  "elongation", None)
#		self.plot_feature_scatter_plot("dist_to_AP_axis", "elongation", None, "Position relative to AP axis", "distance to axis",  "elongation", None)
#		self.plot_feature_scatter_plot("centroid_dist_from_margin", "elongation", None, "Position relative to AP axis", "distance to margin",  "elongation", None)

#		self.plot_feature_scatter_plot("volume_by_res", "sphericity", None, "Position relative to AP axis", "volume",  "sphericity", None)
#		self.plot_feature_scatter_plot("surface_area_by_res", "sphericity", None, "Position relative to AP axis", "surface area",  "sphericity", None)
	

			self.tissue_groups()

			self.plot_groups_at_time_point(0, "flatness", "volume_by_res")
			self.plot_groups_at_time_point(last_time_point, "flatness", "volume_by_res")

			self.plot_groups_at_time_point(0, "elongation", "volume_by_res")
			self.plot_groups_at_time_point(last_time_point, "elongation", "volume_by_res")

			self.plot_group_average_per_time("volume_by_res")
			self.plot_group_average_per_time("elongation")
			self.plot_group_average_per_time("flatness")
			self.plot_group_average_per_time("sphericity")
			self.plot_group_average_per_time("cylindricity")
			self.plot_group_average_per_time("vol_to_hull_vol_ratio")
			self.plot_group_average_per_time("surface_area_by_res")
			self.plot_group_average_per_time("entropy")

			self.groups_in_space(0)
			self.groups_in_space(last_time_point)

			self. plot_shape_hists_per_group()


		self. plot_feature_histograms_over_time("volume_by_res")
		self. plot_feature_histograms_over_time("surface_area_by_res")
		self. plot_feature_histograms_over_time("flatness")
		self. plot_feature_histograms_over_time("elongation")
		self. plot_feature_histograms_over_time("sphericity")
		self. plot_feature_histograms_over_time("entropy")

		#self.plot_feature_scatter_plot("centroid_dist_from_margin", "flatness", None, "Feature by position", "distance to margin",  "flatness", None)
		#self.plot_feature_scatter_plot("centroid_dist_from_margin", "volume_by_res", None, "Feature by position", "distance to margin",  "volume", None)


		self.plot_2dhist_at_timestamp(("centroid_dist_from_margin", "volume_by_res"), 0)
		self.plot_2dhist_at_timestamp(("centroid_dist_from_margin", "volume_by_res"), last_time_point)


		#self.plot_2dhist_at_timestamp(("centroid_dist_from_margin", "flatness"), 0)
		#self.plot_2dhist_at_timestamp(("centroid_dist_from_margin", "flatness"), last_time_point)


		#self.global_feature_plots()




##		self.shape_hist_for_groups_at_time_point(0)
##		self.shape_hist_for_groups_at_time_point(last_time_point)
##		self.plot_dist_to_nucleus_hists()



		pylab.show()		

	def plot_nuclei_at_timestamp(self):
		"""
		3-d plot with all the nuclei color coded from green to blue based on time stamp
		"""

		

		fig = pylab.figure(figsize=(10,3), facecolor='white')
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


		I = self.fetch_slice_at(init_t, init_z)


		f = pylab.figure(figsize = (10,10), facecolor='white')
		f.canvas.set_window_title("Tracklet projections at one slice")
		l1 = pylab.imshow(I, cmap="gray")
		ax1 = pylab.subplot(111)
		pylab.subplots_adjust(bottom=0.25)
		pylab.axis([0,I.shape[1], I.shape[0],0])
		plot_nuclei_with_tracklets(init_t, init_z,ax1 )


		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)
		s_t = Slider(ax_t, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_t)
		f._s_t = s_t

		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z = pylab.axes([0.2, 0.05, 0.65, 0.03], axisbg=axcolor)
		s_z = Slider(ax_z, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z)
		f._s_z = s_z		

		t = init_t
		z = init_z
		z_old = [-1]
		t_old = [-1]


		# call back for time slider
		def update_t(val):

			t = int(s_t.val)
			z = int(s_z.val )

			if (z_old[0] != z) or (t_old[0] !=t):

				I = self.fetch_slice_at(t,z)
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
			z = int(s_z.val)

			if (z_old[0] != z) or (t_old[0] !=t):
				I = self.fetch_slice_at(t,z)
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
		memch = int(CellECT.track_tool.globals.PARAMETER_DICT["membrane-channel"])
		numch = int(CellECT.track_tool.globals.PARAMETER_DICT["number-channels"])
		numz = int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])
		file_index = t* numz * numch + (z)*numch + memch +1
		#print "membrane: ", file_index
		return file_index


	def get_nuclei_file_number_in_tif_sequence(self,t,z):

		nucl_chan = 1
		if int(CellECT.track_tool.globals.PARAMETER_DICT["membrane-channel"]) == 0:
			nucl_chan = 1
		else:
			nucl_chan = 0
		numch = int(CellECT.track_tool.globals.PARAMETER_DICT["number-channels"])
		numz = int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])
		file_index = t* numz * numch + (z)*numch + nucl_chan +1
		#print "nuclei: ", file_index
		return file_index



	def fetch_slice_at(self, t,z):

		t += CellECT.track_tool.globals.PARAMETER_DICT["time-stamps"][0]
	
		membrane_file_location = os.path.join(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"], str(self.get_membrane_file_number_in_tif_sequence(t,z)+1)+".tif")
		membrane_file = sp.misc.imread(membrane_file_location)

		nuclei_file = None
		if CellECT.track_tool.globals.PARAMETER_DICT["number-channels"] > 1:
			nuclei_file_location = os.path.join(CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"], str(self.get_nuclei_file_number_in_tif_sequence(t,z)+1)+".tif")
			nuclei_file = sp.misc.imread(nuclei_file_location)

		if nuclei_file is None:
			return membrane_file
		else:
			I = np.zeros((membrane_file.shape[0], membrane_file.shape[1], 3))
			I[:,:,0] = membrane_file / np.max(membrane_file).astype("float") * 255
			I[:,:,1] = nuclei_file / np.max(nuclei_file).astype("float") * 255
			return I.astype("uint8")

	def fetch_seg_at(self, t,z):

		t += CellECT.track_tool.globals.PARAMETER_DICT["time-stamps"][0]
		filename = 	os.path.join(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"], "timestamp_"+str(t)+"_z_"+ str(z) + "_seg.png")
		Seg = sp.misc.imread(filename)
		return Seg

	def plot_color_tracklets_time_sequence(self, init_time, init_z):
		"""
		Plot a slice of the volume at given t and z, and draw the nuclei color coded
		according to tracklet slider bar to navigate time and z
		"""

		from matplotlib.widgets import Slider

		fig = pylab.figure(figsize =(10,10), facecolor='white')
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

		I = self.fetch_slice_at(t,z)


		#draw_points_at_t_z(init_time, init_z)

		# image subplot
		ax1 = pylab.subplot(111)
		pylab.subplots_adjust(bottom=0.25)
		l1 =  pylab.imshow(I, cmap = "gray")
		pylab.axis([0, I.shape[1], I.shape[0],0])
		draw_points_at_t_z(t,z,ax1)


		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)
		s_t = Slider(ax_t, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_time)
		fig._s_t = s_t

		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z = pylab.axes([0.2, 0.05, 0.65, 0.03], axisbg=axcolor)
		s_z = Slider(ax_z, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z)
		fig._s_z = s_z

		t = init_time
		z = init_z
		z_old = [-1]
		t_old = [-1]

		# call back for time slider
		def update_t(val):

			t = int(s_t.val)
			z = int(s_z.val)

			if (z_old[0] != z) or (t_old[0] !=t):
				I = self.fetch_slice_at(t,z)
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
			z = int(s_z.val)

			if (z_old[0] != z) or (t_old[0] !=t):
				I = self.fetch_slice_at(t,z)
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

		fig = pylab.figure(figsize=(19,8), facecolor='white')
		fig.canvas.set_window_title("Cell lineage visualization")

		from matplotlib.widgets import Slider
		from matplotlib.widgets import Button

		t1 = init_t1
		z1 = init_z1

		t2 = init_t2
		z2 = init_z2

		cc_of_interest = [-1]
		
		cc_dict = connected_components(self.cell_tracker.graph)
		
		I1 = self.fetch_slice_at(t1,z1)
		Seg1 = []
		Seg1.append(self.fetch_seg_at(t1,z1))
		
		Seg2 = []

	
		def make_cell_lineage_mask(tval, target_cc_index, Seg):

			labels = np.unique(Seg)
			target_cc_index = set(target_cc_index)
			
			# make the segment border max value 
			max_val = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].list_of_cell_profiles)
			half_val = int(max_val / 2.)
			mask = np.uint(Seg == 0) * max_val

			for label in labels:
				if label>1:
					cp_index = self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].seg_label_to_cp_list_index[label]
					node_name = "t%d_c%d" % (tval, cp_index) 
					cc_index = cc_dict[node_name]
					
					if cc_index in target_cc_index:
						mask += np.uint(Seg==label) * half_val
						print "--- LINK: Label:", label,"| Tracklet:", cc_index,"| CellProfile:", cp_index, "| Node:", node_name, "| t:",tval
			return mask



		def select_similar_cells(event):

			print "Select similar cells"

			# get cell profile indices of cells of interest
			target_cp_index = [  ]

			cc_of_interest_set = set(cc_of_interest)

			t1 = int(s_t1.val)

			# TODO: no need to search all time series, just search current time
			for node in self.cell_tracker.graph.nodes():

				t,c = self.cell_tracker.get_cell_profile_info_from_node_name(node)

				if t == t1:
					if cc_dict[node] in cc_of_interest_set:
						target_cp_index.append(c)

			similar_cp_index = self.cell_tracker.list_of_cell_profiles_per_timestamp[t1].get_similar(target_cp_index)


			# get the connected components correspondng to these
			for cp_index in similar_cp_index:
				node_name = "t" + str(t1) + "_c" + str(cp_index)
				cc_of_interest.append(cc_dict[node_name])

			# remake selection mask and lineage_mask
	
			

			lineage_mask = make_cell_lineage_mask(int(s_t2.val), cc_of_interest,Seg2[0])
			selection_mask = make_cell_lineage_mask(t1,cc_of_interest, Seg1[0])



			l4.set_data(lineage_mask )
			l21.set_data(selection_mask)
			pylab.draw()




		I2 = self.fetch_slice_at(t2,z2)
		Seg2.append(self.fetch_seg_at(t2,z2))
		lineage_mask = make_cell_lineage_mask(init_t2, cc_of_interest,Seg2[0])
		selection_mask = make_cell_lineage_mask(init_t1,cc_of_interest, Seg1[0])

		
		
		#draw_points_at_t_z(init_time, init_z)


		# image subplot
		ax1 = pylab.subplot(161)
		pylab.subplots_adjust(bottom=0.25)
		l1 =  pylab.imshow(I1, cmap = "gray")
		pylab.axis([0, I1.shape[1], I1.shape[0], 0])

		# segmentation subplot
		ax2 = pylab.subplot(162)
		pylab.subplots_adjust(bottom=0.25)
		l2 =  pylab.imshow(Seg1[0], cmap = self.color_map, vmin = 0, vmax = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[t1].list_of_cell_profiles), picker = True)
		pylab.axis([0, Seg1[0].shape[1], Seg1[0].shape[0], 0])

		# user selection mask
		ax21 = pylab.subplot(163)
		pylab.subplots_adjust(bottom=0.25)
		l21 =  pylab.imshow(selection_mask, cmap = "gist_heat", vmin = 0, vmax = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[t1].list_of_cell_profiles), picker = True)
		pylab.axis([0, Seg1[0].shape[1],  Seg1[0].shape[0], 0])

		# image subplot
		ax3 = pylab.subplot(164)
		pylab.subplots_adjust(bottom=0.25)
		l3 =  pylab.imshow(I2, cmap = "gray")
		pylab.axis([0, I2.shape[1], I2.shape[0], 0])

		# segmentation subplot
		ax31 = pylab.subplot(165)
		pylab.subplots_adjust(bottom=0.25)
		l31 =  pylab.imshow(Seg2[0], cmap = self.color_map,  vmin = 0, vmax = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[t1].list_of_cell_profiles))
		pylab.axis() #[0, Seg2[0].shape[1], 0, Seg2[0].shape[0]])


		# segmentation subplot
		ax4 = pylab.subplot(166)
		pylab.subplots_adjust(bottom=0.25)
		l4 =  pylab.imshow(lineage_mask, cmap = "gist_heat", vmin = 0, vmax = len( self.cell_tracker.list_of_cell_profiles_per_timestamp[t2].list_of_cell_profiles))
		pylab.axis([0, Seg2[0].shape[1], Seg2[0].shape[0], 0])





		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t1 = pylab.axes([0.2, 0.2, 0.25, 0.03], axisbg=axcolor)
		s_t1 = Slider(ax_t1, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_t1)
		fig._s_t1 = s_t1

		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z1 = pylab.axes([0.2, 0.15, 0.25, 0.03], axisbg=axcolor)
		s_z1 = Slider(ax_z1, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z1)
		fig._s_z1 = s_z1

		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t2 = pylab.axes([0.6, 0.2, 0.25, 0.03], axisbg=axcolor)
		s_t2 = Slider(ax_t2, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_t2)
		fig._s_t2 = s_t2

		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z2 = pylab.axes([0.6, 0.15, 0.25, 0.03], axisbg=axcolor)
		s_z2 = Slider(ax_z2, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z2)
		fig._s_z2 = s_z2



		# select similar cells button
		ax_button1 = pylab.axes([0.3, 0.025, 0.4, 0.05])
		button1 = Button(ax_button1, "Select similar cells.")
		button1.on_clicked(select_similar_cells)
		fig._btn = button1



		t1 = init_t1
		z1 = init_z1
		z_old1 = [-1]
		t_old1 = [-1]

		t2 = init_t2
		z2 = init_z2
		z_old2 = [-1]
		t_old2 = [-1]



		# call back for time slider
		def update_t1(val):

			t1 = int(s_t1.val)
			t2 = int(s_t2.val)
			z1 = int(s_z1.val)
			z_seg1 = int(s_z1.val)

			if (z_old1[0] != z1) or (t_old1[0] !=t1):

				I1 = self.fetch_slice_at(t1,z1)
				Seg1[0] = self.fetch_seg_at(t1,z1)

				selection_mask = make_cell_lineage_mask(t1, cc_of_interest, Seg1[0])

				l1.set_data(I1)
				l2.set_data(Seg1[0])
				l21.set_data(selection_mask)

				pylab.draw()

				z_old1[0] = z1
				t_old1[0] = t1



		# call back for z slider
		def update_z1(val):

			t1 = int(s_t1.val)
			t2 = int(s_t2.val)
			z1 = int(s_z1.val)
			z_seg1 = int(s_z1.val)

			if (z_old1[0] != z1) or (t_old1[0] !=t1):

				I1 = self.fetch_slice_at(t1,z1)
				Seg1[0] = self.fetch_seg_at(t1,z1)

				selection_mask = make_cell_lineage_mask(t1, cc_of_interest, Seg1[0])

				l1.set_data(I1)	
				l2.set_data(Seg1[0])
				l21.set_data(selection_mask)

				pylab.draw()
	
				z_old1[0] = z1
				t_old1[0] = t1



		# call back for time slider
		def update_t2(val):

			t2 = int(s_t2.val)
			t1 = int(s_t1.val)
			z2 = int(s_z2.val )
			z_seg2 = int(s_z2.val)

			if (z_old2[0] != z2) or (t_old2[0] !=t2):

				I2 = self.fetch_slice_at(t2,z2)
				Seg2[0] = self.fetch_seg_at(t2,z2)

				lineage_mask = make_cell_lineage_mask(t2, cc_of_interest, Seg2[0])

				l3.set_data(I2)
				l31.set_data(Seg2[0])
				l4.set_data(lineage_mask)

				pylab.draw()

				z_old2[0] = z2
				t_old2[0] = t2



		# call back for z slider
		def update_z2(val):


			t2 = int(s_t2.val)
			t1 = int(s_t1.val)
			z2 = int(s_z2.val )
			z_seg2 = int(s_z2.val)

			if (z_old2[0] != z2) or (t_old2[0] !=t2):
		
				I2 = self.fetch_slice_at(t2,z2)
				Seg2[0] = self.fetch_seg_at(t2,z2)

				lineage_mask = make_cell_lineage_mask(t2, cc_of_interest, Seg2[0])


				l3.set_data(I2)
				l31.set_data(Seg2[0])
				l4.set_data(lineage_mask)

				pylab.draw()
	
				z_old2[0] = z2
				t_old2[0] = t2



		def onpick(event):

			xval = int(event.mouseevent.xdata)
			yval = int(event.mouseevent.ydata)
			tval = int(s_t1.val)
			label = Seg1[0][yval, xval]


			if label == 0:
				print "Border selected."
			if label == 1:
				print "Background selected."

			if label >1:
				cp_index = self.cell_tracker.list_of_cell_profiles_per_timestamp[tval].seg_label_to_cp_list_index[label]

				node_name = "t" + str(tval) + "_c" + str(cp_index)
				if event.mouseevent.button == 1:  
					# left click starts new set
					for i in xrange(0,len(cc_of_interest)):
						cc_of_interest.pop(0)
					cc_of_interest.append(cc_dict[node_name])

				elif event.mouseevent.button == 3:
					# right click adds to the selection set
					cc_name = cc_dict[node_name]
					if not cc_name in cc_of_interest:
						cc_of_interest.append(cc_name)
					else:
						cc_of_interest.remove(cc_name)						


				print cc_of_interest

				print event.mouseevent.button
				if len(cc_of_interest):
					print "SELECTED: Label:", label, "| Tracklet:", cc_of_interest[-1], "| CellProfile:", cp_index, "| Node:", node_name, "| t:", tval

				t2 = int(s_t2.val)
				t1 = int(s_t1.val)
				lineage_mask = make_cell_lineage_mask( t2, cc_of_interest, Seg2[0])
				selection_mask = make_cell_lineage_mask( t1, cc_of_interest, Seg1[0])

				l4.set_data(lineage_mask )
				l21.set_data(selection_mask)
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

		fig = pylab.figure(figsize=(10,10), facecolor='white')
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
					total_tracklets = len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1
					index = min ( tracklet, int(tracklet/ float(total_tracklets) * 255))
					color = pylab.cm.jet(color_idx[index])
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


		I = self.fetch_slice_at(t,z)
		Seg = self.fetch_seg_at(t,z)
		#draw_points_at_t_z(init_time, init_z)
		import time
		time1 = time.time()
		relabel_to_tracklet(Seg, init_time)
		print time.time() - time1

		def onpick(event):

			x = int(event.mouseevent.xdata)
			y = int(event.mouseevent.ydata)
			z = int()
			print Seg[y,x]
		
#		my_colors = make_colors_for_seg(Seg, init_time)

		# image subplot
		ax1 = pylab.subplot(121)
		pylab.subplots_adjust(bottom=0.25)
		l1 =  pylab.imshow(I, cmap = "gray")
		pylab.axis([0, I.shape[1],  I.shape[0], 0])
		draw_points_at_t_z(t,z,ax1)

		# segmentation subplot
		ax2 = pylab.subplot(122)
		pylab.subplots_adjust(bottom=0.25)
		l2 =  pylab.imshow(Seg, vmin = 0, vmax = len(color_idx), cmap = "jet", picker = True)
		pylab.axis([0, Seg.shape[1], Seg.shape[0], 0])




		# slider for time
		axcolor = 'lightgoldenrodyellow'
		ax_t = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)
		s_t = Slider(ax_t, 'time-stamp', 0, len(self.cell_tracker.list_of_cell_profiles_per_timestamp)-1, valinit=init_time)
		fig._s_t = s_t

		# slider for z
		axcolor = 'lightgoldenrodyellow'
		ax_z = pylab.axes([0.2, 0.05, 0.65, 0.03], axisbg=axcolor)
		s_z = Slider(ax_z, 'z-slice', 0, int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])-1, valinit=init_z)
		fig._s_z = s_z

		t = init_time
		z = init_z
		z_old = [-1]
		t_old = [-1]

		# call back for time slider
		def update_t(val):

			t = int(s_t.val)
			z = int(s_z.val / 2) *2
			z_seg = z

			if (z_old[0] != z) or (t_old[0] !=t):
				I = self.fetch_slice_at(t,z)
				l1.set_data(I)
				Seg = self.fetch_seg_at(t,z)
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
			z = int(s_z.val )
			z_seg = z

			if (z_old[0] != z) or (t_old[0] !=t):
				I = self.fetch_slice_at(t,z)
				l1.set_data(I)
				#print "loading ", "seg_all_time_stamps/timestamp_"+str(t)+"_z_"+ str(z) + "_seg.png"
				Seg = self.fetch_seg_at(t,z)
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
		fig.canvas.mpl_connect('pick_event', onpick)


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

		def call_plot_stats(event):
			self.plot_stats()

		def call_plot_tracklets_one_slice(event):
			z_val = int(int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])/2)
			self. plot_tracklets_one_slice(0,z_val)

		def call_plot_color_tracklets_time_sequence(event):
			z_val = int(int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])/2)
			self.plot_color_tracklets_time_sequence(0,z_val)

		def call_cell_lineage_gui(event):
			z_val = int(int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])/2)
			self.cell_lineage_gui(0,1, z_val, z_val)
	
		def call_plot_tracklets_in_slice_with_seg(event):
			z_val = int(int(CellECT.track_tool.globals.PARAMETER_DICT["z-slices-per-stack"])/2)
			self. plot_tracklets_in_slice_with_seg(0,z_val)


		def enter_debug(event):
			pdb.set_trace()
		

		fig = None

		if  CellECT.track_tool.globals.PARAMETER_DICT["with_tracker"]:
			fig = pylab.figure(figsize=(10,7), facecolor='white')
		else:
			fig = pylab.figure(figsize=(10,2), facecolor='white')

		fig.canvas.set_window_title("Visualize Results")




		if CellECT.track_tool.globals.PARAMETER_DICT["with_tracker"]:	


			ax_button1 = pylab.axes([0.075, 0.85, 0.6, 0.075])
			button1 = Button(ax_button1, "3-D plot of nuclei color coded by timestamp")
			button1.on_clicked(call_plot_nuclei_at_timestamp)


			ax1 = pylab.axes( [0.725, 0.85, 0.2, 0.075])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "01.png"))
			pylab.imshow(im)
			ax1.axis("off")

		

			ax_button3 = pylab.axes([0.075, 0.75, 0.6, 0.075])
			button3 = Button(ax_button3, "Statistical plots")
			button3.on_clicked(call_plot_stats)

			ax3 = pylab.axes( [0.725, 0.75, 0.2, 0.075])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir,"03.png"))
			pylab.imshow(im)
			ax3.axis("off")



			ax_button2 = pylab.axes([0.075, 0.65, 0.6, 0.075])
			button2 = Button(ax_button2, "3-d plot of nuclei color coded by tracklet")
			button2.on_clicked(call_plot_nuclei_at_timestamp_with_tracklet_coloring)

			ax2 = pylab.axes([0.725, 0.65, 0.2, 0.075] )
			im = sp.misc.imread(os.path.join(self.thumbnail_dir,"02.png"))
			pylab.imshow(im)
			ax2.axis("off")




			ax_button4 = pylab.axes([0.075, 0.55, 0.6, 0.075])
			button4 = Button(ax_button4, "Draw track projections at one slice")
			button4.on_clicked(call_plot_tracklets_one_slice)

			ax4 = pylab.axes( [0.725, 0.55, 0.2, 0.075])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "04.png"))
			pylab.imshow(im)
			ax4.axis("off")




			ax_button5 = pylab.axes([0.075, 0.45, 0.6, 0.075])
			button5 = Button(ax_button5, "Plot tracklets as color coded nuclei at slice")
			button5.on_clicked(call_plot_color_tracklets_time_sequence)
	
			ax5 = pylab.axes( [0.725, 0.45, 0.2, 0.075])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "05.png"))
			pylab.imshow(im)
			ax5.axis("off")





			ax_button6 = pylab.axes([0.075, 0.35, 0.6, 0.075])
			button6 = Button(ax_button6, "Cell lineage visualization")
			button6.on_clicked(call_cell_lineage_gui)

			ax6 = pylab.axes( [0.725, 0.35, 0.2, 0.075])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "06.png"))
			pylab.imshow(im)
			ax6.axis("off")




			ax_button7 = pylab.axes([0.075, 0.25, 0.6, 0.075])
			button7 = Button(ax_button7, "Segmentation color coded by tracklets")
			button7.on_clicked(call_plot_tracklets_in_slice_with_seg)

			ax7 = pylab.axes( [0.725, 0.25, 0.2, 0.075])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "07.png"))
			pylab.imshow(im)
			ax7.axis("off")



			ax_button8 = pylab.axes([0.075, 0.05, 0.6, 0.075])
			button8 = Button(ax_button8, "Enter debug...")
			button8.on_clicked(enter_debug)

			ax8 = pylab.axes( [0.725, 0.05, 0.2, 0.075])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "08.png"))
			pylab.imshow(im)
			ax8.axis("off")


		else:
			ax_button1 = pylab.axes([0.075, 0.55, 0.6, 0.35])
			button1 = Button(ax_button1, "3-D plot of nuclei color coded by timestamp")
			button1.on_clicked(call_plot_nuclei_at_timestamp)


			ax1 = pylab.axes( [0.725, 0.55, 0.2, 0.35])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "01.png"))
			pylab.imshow(im)
			ax1.axis("off")

		

			ax_button3 = pylab.axes([0.075, 0.15, 0.6, 0.35])
			button3 = Button(ax_button3, "Statistical plots")
			button3.on_clicked(call_plot_stats)

			ax3 = pylab.axes( [0.725, 0.15, 0.2, 0.35])
			im = sp.misc.imread(os.path.join(self.thumbnail_dir, "03.png"))
			pylab.imshow(im)
			ax3.axis("off")


		pylab.show()
		

