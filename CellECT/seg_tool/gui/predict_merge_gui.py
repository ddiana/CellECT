import pylab
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import pdb
import numpy as np
from matplotlib.widgets import CheckButtons

class MergePredictorUI(object):

	def __init__(self,):


		self.vol = None
		self.label_map = None
		self.highlight_map = None

		self.answer = False
		self.suggest_another = True

		self.label1 = None
		self.label2 = None
		self.list_to_merge = []
		self.list_to_delete = []


	def set_data(self, vol, vol_nuclei, label_map, highlight_map, color_map, label1, label2, list_to_merge, list_to_delete, all_answers, score):

		self.vol = vol
		self.vol_nuclei = vol_nuclei
		self.label_map = label_map
		self.highlight_map = highlight_map
		self.color_map = color_map
		self.label1 = label1
		self.label2 = label2
		self.list_to_merge = list_to_merge
		self.list_to_delete = list_to_delete
		self.all_answers = all_answers
		self.score = score


	def merge_callback(self,event):
		self.answer = True
		print "Yes, merge!"
		self.list_to_merge.append((self.label1, self.label2))
		self.all_answers. append((self.score,True))
		pylab.close(self.fig)

	def delete_red_callback(self,event):
		self.answer = True
		print "Delete red!"
		self.list_to_delete.append((self.label2))
		self.all_answers. append((self.score,True))
		pylab.close(self.fig)

	def delete_yellow_callback(self,event):
		self.answer = True
		print "Delete yellow!"
		self.list_to_delete.append((self.label1))
		self.all_answers. append((self.score,True))
		pylab.close(self.fig)

	def dont_merge_callback(self, event):
		self.answer = False
		print "No, dont' merge!"
		self.all_answers.append((self.score,False))
		pylab.close(self.fig)

	def vol_slice(self,z):
		slice_to_show = np.zeros((self.vol.shape[0], self.vol.shape[1], 3))
		slice_to_show[:,:,0] = self.vol[:,:,z]
		slice_to_show[:,:,0] = slice_to_show[:,:,0].astype("float")/np.max(slice_to_show[:,:,0])*255
		if not self.vol_nuclei is None:
			slice_to_show[:,:,1] = self.vol_nuclei[:,:,z]
			slice_to_show[:,:,1] = slice_to_show[:,:,1].astype("float")/np.max(slice_to_show[:,:,1])*255

		return slice_to_show.astype("uint8")



	def display(self):

		self.fig = pylab.figure(facecolor='white')
		self.fig.canvas.set_window_title("Should merge these two segments?")



		z0 = self.vol.shape[2] /2

		ax1 = pylab.subplot(131)
		pylab.subplots_adjust(bottom=0.25)
		slice_to_show = self.vol_slice(z0)
		l1 =  pylab.imshow(slice_to_show, interpolation="nearest")   #cax = l2
		pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
		ax1.set_title("Confocal Volume")

		ax2 = pylab.subplot(132)
		pylab.subplots_adjust(bottom=0.25)
		min_var_cmap_seg = self.label_map.min()
		max_var_cmap_seg = self.label_map.max()
		l2 =  pylab.imshow(self.label_map[:,:,z0], interpolation="nearest", cmap = self.color_map, vmin= min_var_cmap_seg, vmax = max_var_cmap_seg)   #cax = l2
		pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
		ax2.set_title("Segmentation")	

		ax3 = pylab.subplot(133)
		pylab.subplots_adjust(bottom=0.25)
		min_var_cmap_hl = 0
		max_var_cmap_hl = 255
		l3 =  pylab.imshow(self.highlight_map[:,:,z0], interpolation="nearest", cmap = "hot", vmin= min_var_cmap_hl, vmax = max_var_cmap_hl)   #cax = l2
		pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
		ax3.set_title("Should merge?")	
		

		axcolor = 'lightgoldenrodyellow'
		ax_z = pylab.axes([0.2, 0.20, 0.65, 0.03], axisbg=axcolor)

		def update(event):

			z = s_z.val

			slice_to_show = self.vol_slice(z)


			l1.set_data(slice_to_show.astype("uint8"))
			l2.set_data(self.label_map[:,:,z])
			l3.set_data(self.highlight_map[:,:,z])

			pylab.draw()





		a_merge = pylab.axes([0.1, 0.07, 0.12, 0.07])
		self.b_merge = Button(a_merge, "Merge")
		self.b_merge.on_clicked(self.merge_callback)

		a_delete_red = pylab.axes([0.23, 0.07, 0.16, 0.07])
		self.b_delete_red = Button(a_delete_red, "Delete red")
		self.b_delete_red.on_clicked(self.delete_red_callback)

		a_delete_yellow = pylab.axes([0.40, 0.07, 0.22, 0.07])
		self.b_delete_yellow = Button(a_delete_yellow, "Delete yellow")
		self.b_delete_yellow.on_clicked(self.delete_yellow_callback)


		a_dont_merge= pylab.axes([0.63, 0.07, 0.3, 0.07])
		self.b_dont_merge = Button(a_dont_merge, "Don't merge / Don't know")
		self.b_dont_merge.on_clicked(self.dont_merge_callback)



#		rax = pylab.axes([0.4, 0.00, 0.15, 0.15])
#		rax.set_axis_off()
#		self.check = CheckButtons(rax, ['Suggest another'],  [True])

#		def set_suggest_another(label):
#			print label
#			#self.suggest_another = not self.suggest_another
#			pylab.draw()

#		self.check.on_clicked(set_suggest_another)

		s_z = Slider(ax_z, 'z-slice', 0, self.vol.shape[2]-1, valinit=z0)
		s_z.on_changed(update)
		pylab.show()

		


