# Colour segmentation class
# To run, use the following steps
# One image: make sure the 'one image' part at the bottom of the script is uncommented. Then "python colourseg_class.py inputimage.tiff"
# Folder of images: make sure the folder image part at the bottom of script is uncommented. Then "python colourseg_class.py imagefolder"
# Dependencies: Tkinter, PIL, matplotlib, skimage, numpy, pandas, os, argv
from PIL import Image, ImageOps, ImageTk, ImageChops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, gaussian
from skimage import exposure, color
import os
import numpy as np
from sys import argv

# Inputs cause I don't know how to use argparse :(
script, input_folder = argv

# Global variable used for selecting plots from matplotlib figures. Turns into string, so it doesn't use much ram
current_axis = None

class ImageObject:
	def __init__(self, image):

		# List of colour points from the function 'select_points'
		colour_data = []
		img = Image.open(image)
		# PIL image
		self.pilimg = img
		# Colour numpy image - 3d numpy array where dim1 is columns, dim2 is rows, dim3 is r, g, b
		self.npimg = np.array(img)
		img = img.convert('L')
		img = ImageOps.invert(img)

		img = np.asarray(img)
		img.setflags(write=True)
		img[img == 255] = 0
		# Grayscale numpy image. Inverted because finding cells requires maxima
		# 2d array where dim1 is columns, dim2 is rows
		self.gray_npimg = img
		head, tail = os.path.split(image)

		# Image name (head is name, tail is path) for file saving purposes
		self.img_name = tail
		self.img_path = head

	"""
	Thresholding. Thresholds automatically, then calls itself again to specify which threshold to use.
	Uses NUMPY IMAGE

	There are three types of thresholding inputs:
		1. Broad - User has no idea but wants to view entire spectrum and choose from there
		2. Automatic - User has no idea, wants script to choose rough area so they can specify
		3. Manual - User knows threshold range, chooses which level is optimal
	"""
	def threshold(self, image=None, threshold=None, ret=None):
		if image == None:
			image = self.gray_npimg
		image.setflags(write=True)

		fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		ax1, ax2, ax3, ax4, ax5, ax6 = ax.ravel()
		ax1.set_title('Original')
		ax2.set_title('Threshold 1')
		ax3.set_title('Threshold 2')
		ax4.set_title('Threshold 3')
		ax5.set_title('Threshold 4')
		ax6.set_title('Threshold 5')

		# Peak local max finds local maxima. These are saved as cell pixels (maxima_pixels)
		if threshold == None:
			coordinates1 = peak_local_max(image, min_distance=0, threshold_rel=(0.1), exclude_border=False)
			coordinates2 = peak_local_max(image, min_distance=0, threshold_rel=(0.3), exclude_border=False)
			coordinates3 = peak_local_max(image, min_distance=0, threshold_rel=(0.5), exclude_border=False)
			coordinates4 = peak_local_max(image, min_distance=0, threshold_rel=(0.7), exclude_border=False)
			coordinates5 = peak_local_max(image, min_distance=0, threshold_rel=(0.9), exclude_border=False)
			thresh_dictionary = { str(ax2.title): 0.1, str(ax3.title): 0.3, str(ax4.title): 0.5, str(ax5.title): 0.7, str(ax6.title): 0.9 }
		elif threshold == 'auto' or threshold == 'automatic':
			otsu = float(threshold_otsu(image))/1000
			coordinates1 = peak_local_max(image, min_distance=0, threshold_rel=(otsu-0.2), exclude_border=False)
			coordinates2 = peak_local_max(image, min_distance=0, threshold_rel=(otsu-0.1), exclude_border=False)
			coordinates3 = peak_local_max(image, min_distance=0, threshold_rel=(otsu), exclude_border=False)
			coordinates4 = peak_local_max(image, min_distance=0, threshold_rel=(otsu+0.1), exclude_border=False)
			coordinates5 = peak_local_max(image, min_distance=0, threshold_rel=(otsu+0.2), exclude_border=False)
			thresh_dictionary = { str(ax2.title): (otsu-0.2), str(ax3.title): (otsu-0.1), str(ax4.title): (otsu), str(ax5.title): (otsu+0.1), str(ax6.title): (otsu+0.2) }
		else:
			step_size = (threshold[1] - threshold[0])/4
			coordinates1 = peak_local_max(image, min_distance=0, threshold_rel=(threshold[0]), exclude_border=False)
			coordinates2 = peak_local_max(image, min_distance=0, threshold_rel=(threshold[0]+1*step_size), exclude_border=False)
			coordinates3 = peak_local_max(image, min_distance=0, threshold_rel=(threshold[0]+2*step_size), exclude_border=False)
			coordinates4 = peak_local_max(image, min_distance=0, threshold_rel=(threshold[0]+3*step_size), exclude_border=False)
			coordinates5 = peak_local_max(image, min_distance=0, threshold_rel=(threshold[0]+4*step_size), exclude_border=False)
			thresh_dictionary = { str(ax2.title): threshold[0], str(ax3.title): (threshold[0]+1*step_size), str(ax4.title): (threshold[0]+2*step_size), str(ax5.title): (threshold[0]+3*step_size), str(ax6.title): (threshold[0]+4*step_size) }

		# Display results
		ax1.imshow(self.npimg, cmap=plt.cm.gray)
		ax1.axis('on')

		ax2.imshow(self.npimg, cmap=plt.cm.gray)
		ax2.autoscale(False)
		ax2.plot(coordinates1[:, 1], coordinates1[:, 0], 'r.'),
		ax2.axis('on')

		ax3.imshow(self.npimg, cmap=plt.cm.gray)
		ax3.autoscale(False)
		ax3.plot(coordinates2[:, 1], coordinates2[:, 0], 'r.'),
		ax3.axis('on')

		ax4.imshow(self.npimg, cmap=plt.cm.gray)
		ax4.autoscale(False)
		ax4.plot(coordinates3[:, 1], coordinates3[:, 0], 'r.'),
		ax4.axis('on')

		ax5.imshow(self.npimg, cmap=plt.cm.gray)
		ax5.autoscale(False)
		ax5.plot(coordinates4[:, 1], coordinates4[:, 0], 'r.'),
		ax5.axis('on')

		ax6.imshow(self.npimg, cmap=plt.cm.gray)
		ax6.autoscale(False)
		ax6.plot(coordinates5[:, 1], coordinates5[:, 0], 'r.'),
		ax6.axis('on')

		ax_dictionary = {str(ax2.title): coordinates1, str(ax3.title): coordinates2, str(ax4.title): coordinates3, str(ax5.title): coordinates4, str(ax6.title): coordinates5}

		#function to be called when mouse is clicked
		def axes_change(event):
			global current_axis
			current_axis = event.inaxes.title
		def axes_select(event):
			if event.button == 3:
				plt.close()
		cid = fig.canvas.mpl_connect('axes_enter_event', axes_change)
		cid = fig.canvas.mpl_connect('button_release_event', axes_select)
		plt.show()


		threshold2 = thresh_dictionary[str(current_axis)]

		if ret == "npmaxima":
			self.maxima_pixels = peak_local_max(image, min_distance=0, threshold_rel=threshold2, indices=False, exclude_border=False)
			return self.maxima_pixels
		else:
			self.maxima_pixels = self.threshold(threshold=[threshold2-0.1, threshold2+0.1], ret="npmaxima")
			return self.maxima_pixels

	# Select_points lets you select pixels on the image and use them for colour segmentation.
	# Image will automatically be loaded as the pil image
	# Diameter is how big the dot you want to be on the image. 
	# USES PIL IMAGE
	def select_points(self, image=None, diameter=6):
		if image == None:
			image = self.npimg
		diff = int(diameter/2) # Radius
		divide = (2*int(diameter/2))**2 # What to divide by when adding to r/g/b lists

		# Starting to implement "lab" colour space
		#lab = color.rgb2lab(imc)
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.imshow(image, cmap=plt.cm.gray)


		r_list = []
		g_list = []
		b_list = []
		locations = []
	    #function to be called when mouse is clicked
		def onclick(event):
			if event.button == 3:
				ix, iy = int(event.xdata), int(event.ydata)
				print ix, iy
				locations.append( (ix, iy) )
				# Add dots to matplotlib (doesn't work perfectly)
				ax1.add_patch(patches.Circle((ix, iy), radius=diff, color='black'))
				avr = 0
				avg = 0
				avb = 0
				for a in range(0-diff, diff):
					for b in range(0-diff, diff):
						avr += image[iy+b, ix+a][0]
						avg += image[iy+b, ix+a][1]
						avb += image[iy+b, ix+a][2]
				r_list.append( avr / divide )
				g_list.append( avg / divide )
				b_list.append( avb / divide )

		cid = fig1.canvas.mpl_connect('button_release_event', onclick)
		plt.show()
		
		self.colour_data = [r_list, g_list, b_list]
		return self.colour_data

	# Analyze_numpy takes r/g/b and sorts them based on click data (data) that you give it, along with spread.
	def analyze_numpy(self, r, g, b, data, npmaxima, spread, ret='plot'):
		# C = colour for below
		# (C <     MAX       +      STANDARD DEV)       & (C >      MIN      -     STANDARD DEV)        & IT'S A CELL PIXEL
		r[(r < (data['rmax'] + spread*data['r_stdev'])) & (r > (data['rmin'] - spread*data['r_stdev'])) & npmaxima == True] = 0
		g[(g < (data['gmax'] + spread*data['g_stdev'])) & (g > (data['gmin'] - spread*data['g_stdev'])) & npmaxima == True] = 0
		b[(b < (data['bmax'] + spread*data['b_stdev'])) & (b > (data['bmin'] - spread*data['b_stdev'])) & npmaxima == True] = 0
		if ret == 'plot':
			return np.dstack( (r, g, b) )
		else:
			labeled_pixels = np.zeros( (r.shape), dtype=bool )
			labeled_pixels[(r == 0) & (g == 0) & (b == 0) & npmaxima == True] = True
			return np.count_nonzero(labeled_pixels)
	# Segment2 lets the user control the colour segmentation part more
		# Image is the RAW PIL IMAGE.
		# npmaxima is the segmented cells. This is a 2d array (w x h)\
		# Colours is the result of select_points. It is necessary for the "colour" part of segmentation
	def segment2(self, npmaxima, colours, image=None):
		if image == None:
			try:
				imc = self.npimg
				image = self.pilimg
			except:
				imc = self.npimg
				image = self.pilimg
		imc.setflags(write=True)
		red, green, blue = image.split()
		r = np.asarray(red)
		g = np.asarray(green)
		b = np.asarray(blue)
		r.setflags(write=True)
		g.setflags(write=True)
		b.setflags(write=True)

		print "Recorded " + str(len(colours[1])) + " points. Analyzing image..."

		# Record relevant colour info in a dictionary called inp
		inp = {}
		inp['r_stdev'] = np.std(colours[0])
		inp['g_stdev'] = np.std(colours[1])
		inp['b_stdev'] = np.std(colours[2])
		inp['rmax'] = max(colours[0])
		inp['rmin'] = min(colours[0])
		inp['gmax'] = max(colours[1])
		inp['gmin'] = min(colours[1])
		inp['bmax'] = max(colours[2])
		inp['bmin'] = min(colours[2])
		cellpixel_count = float(np.sum(npmaxima))

		# Generate several options for user to pick from\
		im1 = self.analyze_numpy(r, g, b, inp, npmaxima,   0)
		im2 = self.analyze_numpy(r, g, b, inp, npmaxima, 0.5)
		im3 = self.analyze_numpy(r, g, b, inp, npmaxima,   1)
		im4 = self.analyze_numpy(r, g, b, inp, npmaxima, 1.5)
		im5 = self.analyze_numpy(r, g, b, inp, npmaxima,   2)

		# Making the figs using im1-im5 and of course including the original image (imc)
		fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		ax1, ax2, ax3, ax4, ax5, ax6 = ax.ravel()

		ax1.imshow(imc, cmap=plt.cm.gray)
		ax1.axis('on')
		ax1.set_title('Original')

		ax2.imshow(im1, cmap=plt.cm.gray)
		ax2.autoscale(False)
		ax2.axis('on')
		ax2.set_title('Threshold 1')

		ax3.imshow(im2, cmap=plt.cm.gray)
		ax3.autoscale(False)
		ax3.axis('on')
		ax3.set_title('Threshold 2')

		ax4.imshow(im3, cmap=plt.cm.gray)
		ax4.autoscale(False)
		ax4.axis('on')
		ax4.set_title('Threshold 3')

		ax5.imshow(im4, cmap=plt.cm.gray)
		ax5.autoscale(False)
		ax5.axis('on')
		ax5.set_title('Threshold 4')

		ax6.imshow(im5, cmap=plt.cm.gray)
		ax6.autoscale(False)
		ax6.axis('on')
		ax6.set_title('Threshold 5')

		ax_dictionary = {str(ax2.title): 0, str(ax3.title): 0.5, str(ax4.title): 1, str(ax5.title): 1.5, str(ax6.title): 2}

		#functions to be called when mouse is clicked
		def axes_change(event):
			global current_axis
			current_axis = event.inaxes.title
		def axes_select(event):
			if event.button == 3:
				plt.close()
		cid = fig.canvas.mpl_connect('axes_enter_event', axes_change)
		cid = fig.canvas.mpl_connect('button_release_event', axes_select)
		plt.show()

		threshold2 = ax_dictionary[str(current_axis)]
		proteinpixel_count = self.analyze_numpy(r, g, b, inp,npmaxima, threshold2, ret='bla')
		print str(100*proteinpixel_count/cellpixel_count) + "% of pixels are labeled"
		return float(100*proteinpixel_count/cellpixel_count)

	def saveimage(self, image):
		image.save("Labeled-" + self.tail + ".png")
#############################################################################################################
# Use this for one image
imge = ImageObject(input_folder)
imge.segment2(imge.threshold(), imge.select_points())

#############################################################################################################
# # Use this for a folder
# for subdir, dirs, files in os.walk(input_folder):
# 	for x in files:
# 		print "Processing " + str(x)
# 		imge = ImageObject(input_folder + x)
# 		percent = imge.segment2(imge.threshold_spec(), imge.select_points())


# print percentlist

""" 
Notes:
	- Implement clickable graphs
	- Collect statistical info for comparison (csv?)
		- Image stats
		- Colour selection and thresholds

Texture
Ilastik uses StructureTensorEigenvalues and HessianOfGaussianEigenvalues
"""