# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import pylab
from matplotlib.widgets import Slider

"""
Tools to display a 3-D volume and slide through it.
"""

def display_volume(vol):

	"Display 3-d volume, slice by slicer, with slider to traverse it."

	from matplotlib.widgets import Slider
	


	ax = pylab.subplot(111)
	pylab.subplots_adjust(bottom=0.25)

	z0 = int(np.floor(vol.shape[2]/2))


	l =  pylab.imshow(vol[:,:,z0], interpolation="nearest", cmap = "gist_heat")
	#pylab.axis([0, vol.shape[0], 0, vol.shape[1]])

	axcolor = 'lightgoldenrodyellow'
	ax_z = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)

	s_z = Slider(ax_z, 'z', 0, vol.shape[2]-1, valinit=z0)

	def update(val):
		z = s_z.val
		l.set_data(vol[:,:,z])
		pylab.draw()
	
	s_z.on_changed(update)
	pylab.show()
		


def display_volume_two(vol1, vol2, z_default = -1):

	"Slice by slice display of 3-D volume with slider. Two volumes displayed side by side."

	from matplotlib.widgets import Slider

	fig = pylab.figure()

	if z_default > -1:
		z0 = z_default
	else:
		z0 = int(np.floor(vol1.shape[2]/2))

	ax1 = pylab.subplot(121)
	pylab.subplots_adjust(bottom=0.25)
	l1 =  pylab.imshow(vol1[:,:,z0], interpolation="nearest", cmap = "gist_heat")
	#pylab.axis([0, vol1.shape[0], 0, vol1.shape[1]])

	ax2 = pylab.subplot(122)
	pylab.subplots_adjust(bottom=0.25)
	l2 =  pylab.imshow(vol2[:,:,z0], interpolation="nearest", cmap = "spectral")
	#pylab.axis([0, vol2.shape[0], 0, vol2.shape[1]])

	axcolor = 'lightgoldenrodyellow'
	ax_z = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)

	s_z = Slider(ax_z, 'z-slice', 0, vol1.shape[2]-1, valinit=z0)

	def update(val):
		z = s_z.val
		l1.set_data(vol1[:,:,z])
		l2.set_data(vol2[:,:,z])
		pylab.draw()
	
	s_z.on_changed(update)
	pylab.show()
	
	
	
		
	
def display_volume_two_get_clicks(vol1, vol2, z_default = -1):

	"""
	Display two voluemes, side by side, with slider to traverse them.
	Also picker function to display value at mouse click.	
	"""


	fig = pylab.figure(figsize=(17,16))
	
	
	class MouseEvent:
		def __init__(self, button, xval, yval, zval):
			self.button = button # left or right
			self.xval = xval
			self.yval = yval
			self.zval = zval
			
		def setInfo(self, button, xval, yval,zval):
			self.button = button # left or right
			self.xval = xval
			self.yval = yval
			self.zval = zval
	

	if z_default > -1:
		z0 = z_default
	else:
		z0 = int(np.floor(vol1.shape[2]/2))

	ax1 = pylab.subplot(121)
	pylab.subplots_adjust(bottom=0.25)
	l1 =  pylab.imshow(vol1[:,:,z0], interpolation="nearest", cmap = "gist_heat")  
	pylab.axis()#[0, vol1.shape[0], 0, vol1.shape[1]])

	ax2 = pylab.subplot(122)
	pylab.subplots_adjust(bottom=0.25)
	l2 =  pylab.imshow(vol2[:,:,z0], interpolation="nearest",  cmap = "gist_heat", picker = True)   #cax = l2
	pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])

	axcolor = 'lightgoldenrodyellow'
	ax_z = pylab.axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)

	s_z = Slider(ax_z, 'z-slice', 0, vol1.shape[2]-1, valinit=z0)

	def update(val):
		z = s_z.val
		l1.set_data(vol1[:,:,z])
		l2.set_data(vol2[:,:,z])
		pylab.draw()
	
	s_z.on_changed(update)
	
	merge_mouse_event1 = MouseEvent(-2,0,0,0)
	merge_mouse_event2 = MouseEvent(-2,0,0,0)
	split_mouse_event  = MouseEvent(-2,0,0,0)
	
	
	def onpick(event):

		xval = event.mouseevent.xdata
		yval = event.mouseevent.ydata
		zval = s_z.val
		
		if event.mouseevent.button != 1:
			print "To merge: ", "x =", int(yval), "y =", int(xval), "z =", int(zval)
			if merge_mouse_event1.xval == 0:
				merge_mouse_event1.setInfo(event.mouseevent.button, int(yval), int(xval), int(zval))			
			else:
				merge_mouse_event2.setInfo(event.mouseevent.button, int(yval), int(xval), int(zval))
			
		else:
			print "To split: ", "x =", int(yval), "y =", int(xval), "z =", int(zval)
			split_mouse_event.setInfo(event.mouseevent.button, int(yval), int(xval), int(zval))
		# note x,y values switched for gui
		
#		if event.mouseevent.button != 1:
#			# always remember the last left node click, if current click is right, the print edge with last left
#			# leftNodeFromMouse.lastLeftNode = nodeName
#		#else:
#			if lastMouseEvent.button == 1 and lastMouseEvent.button !=-2 :
#				print yval, xval
				# note x,y values switched for gui
				
		#lastMouseEvent.setInfo(event.mouseevent.button, xval, yval)
	
	
	fig.canvas.mpl_connect('pick_event', onpick)
	pylab.show()
	
	
	return split_mouse_event, merge_mouse_event1, merge_mouse_event2
	
	
	
