import pylab


class MergePredictorUI(object):

	def __init__(self,)

		self.vol = None
		self.label_map = None
		self.highlight_map = None

		self.answer = False


	def set_data(self, vol, label_map, highlight_map)

		self.vol = vol
		self.label_map = label_map
		self.highlight_map = highlight_map


	def merge_callback(self,event):
		self.answer = True

	def dont_merge_callback(self, event):
		self.answer = False


	def display(self):

		self.fig = pylab.figure()

		a_merge = pylab.axes([0.10, 0.05, 0.08, 0.05])
		a_dont_merge= pylab.axes([0.20, 0.05, 0.16, 0.05])

	
		b_merge = Button(a_merge, 'Yes, merge.')
		b_merge.on_clicked(self.merge_callback)
		b_dont_merge = Button(a_dont_merge, "Don't merge / Don't know")
		b_dont_merge.on_clicked(self.dont_merge_callback)

		z0 = vol.shape[2] /2

		ax1 = pylab.subplot(131)
		pylab.subplots_adjust(bottom=0.25)
		min_var_cmap_vol = self.vol.min()
		max_var_cmap_vol = self..max()
		l2 =  pylab.imshow(vol[:,:,z0], interpolation="nearest", cmap = "PRGn", vmin= min_var_cmap_vol, vmax = max_var_cmap_vol)   #cax = l2
		pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
		ax2.set_title("Confocal Volume")

		ax1 = pylab.subplot(132)
		pylab.subplots_adjust(bottom=0.25)
		min_var_cmap_vol = self.vol.min()
		max_var_cmap_vol = self..max()
		l2 =  pylab.imshow(vol[:,:,z0], interpolation="nearest", cmap = "PRGn", vmin= min_var_cmap_vol, vmax = max_var_cmap_vol)   #cax = l2
		pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
		ax2.set_title("Confocal Volume")	
		


