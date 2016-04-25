# Learn image
# Rohan Borkar, April 2016

# Dependencies: PIL, matplotlib, skimage, sklearn, numpy
from PIL import Image, ImageOps, ImageChops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import svm
from skimage import color
from skimage.measure import label, regionprops
from skimage.feature import structure_tensor, structure_tensor_eigvals
import os
import numpy as np
from sys import argv

script, image = argv

class LearnImage:
	def __init__(self, image):
		# Ask how many clusters to segment the image into
		self.segs = input("Enter number of clusters: ")
		self.clusters = []
		self.digits = []

		# Load image (npimg), and record width, height, directory and image name
		npimg = Image.open(image)
		self.npimg = np.asarray(npimg)
		self.imgwidth = self.npimg.shape[0]
		self.imgheight = self.npimg.shape[1]
		self.head, self.tail = os.path.split(image)

		r = self.npimg[:,:,0]
		g = self.npimg[:,:,1]
		b = self.npimg[:,:,2]
		# Structure tensor texture arrays
		Axx, Axy, Ayy = structure_tensor(r, sigma=0.1)
		r_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		Axx, Axy, Ayy = structure_tensor(g, sigma=0.1)
		g_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		Axx, Axy, Ayy = structure_tensor(b, sigma=0.1)
		b_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		tensor = np.stack( (r_tensor[0], g_tensor[0], b_tensor[0]), axis=2 )
		# Aggregate all the R/G/B, LAB, and texture arrays together. Each time a point is selected, the script will record 9 data values for each pixel
		self.features = np.concatenate((self.npimg, color.rgb2lab(self.npimg), tensor), axis=2)
		print "Finished importing and processing image"

	# Select points comes from my old script, it allows the user to right click on points on the image to record them as training data
	# NEVER call this script, it is called by "collect_data"
	def select_points(self, npimage, rgblab_img, cluster):
		diameter = 6 # Diameter of circle to draw and record when points are selected
		diff = int(diameter/2) # Radius
		divide = (2*int(diameter/2))**2 # What to divide by when adding to r/g/b lists
		# Starting to implement "lab" colour space
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.imshow(npimage, cmap=plt.cm.gray)
		plt.title("Select at least twenty points for cluster " + str(cluster+1))

	    #function to be called when mouse is clicked
		def onclick(event):
			if event.button == 3:
				ix, iy = int(event.xdata), int(event.ydata)
				ax1.add_patch(patches.Circle((ix, iy), radius=diff, color='black'))
				for a in range(0-diff, diff):
						for b in range(0-diff, diff):
							# Add surrounding pixel data along with specific label
							self.clusters.append( rgblab_img[iy+b, ix+a] )
							self.digits.append(cluster)
		cid = fig1.canvas.mpl_connect('button_release_event', onclick)
		plt.show()

	# Calls select_points for each cluster the user has specified. Collects training data.
	def collect_data(self):# Either collect training data
		for x in range(0, self.segs):
			self.select_points(self.npimg, self.features, x)
		print "Finished collecting training data"

	# Training the model
	def model(self):
		print "Fitting data..."
		# Arrange training data into np arrays
		self.clusters = np.asarray(self.clusters)
		self.digits = np.asarray(self.digits)
		# Train (fit) the model on user input data
		self.clf = svm.SVC(gamma=0.001, C=100.)
		self.clf.fit(self.clusters, self.digits)

	# Predict the entire image based on the model 
	def analyze_image(self):
		print "Predicting image..."
		self.features = np.reshape(self.features, ((self.imgwidth*self.imgheight), 9))
		self.predicted_img = self.clf.predict(self.features)
		self.predicted_img = np.array(self.predicted_img)
		self.predicted_img = np.reshape(self.predicted_img, (self.imgwidth, self.imgheight))
		self.features = np.reshape(self.predicted_img, (self.imgwidth, self.imgheight))

		# Make a plot to represent the image
		fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		ax1, ax2 = ax.ravel()
		ax1.imshow(self.npimg, cmap=plt.cm.gray)
		ax2.imshow(self.predicted_img, cmap=plt.cm.gray)
		plt.show()
		print "Labeling data..."

	def mask(self):
		# Labels takes the results of "predict image" and creates a separate object for each set of connected pixels
		labels = label(self.predicted_img)

		# Optional sorting objects based on size.
		# properties = regionprops(labels)
		# for x in properties:
		# 	if x['area'] < 100:
		# 		labels[labels == x['label']] = 0

		# Separate into r/g/b arrays for masking. This may not be the most optimal method, still need to look into it
		r = self.npimg[:,:,0]
		g = self.npimg[:,:,1]
		b = self.npimg[:,:,2]
		r.setflags(write=True)
		g.setflags(write=True)
		b.setflags(write=True)
		r[labels == 0] = 0
		g[labels == 0] = 0
		b[labels == 0] = 0

		self.masked = np.stack( (r, g, b), axis=2 )

	def save(self):
		plt.imsave(fname="Masked_" + self.tail, arr=self.masked)
		print "Mask has been saved."

	def run(self):
		self.collect_data()
		self.model()
		self.analyze_image()
		self.mask()
		self.save()

img = LearnImage(image)
img.run()