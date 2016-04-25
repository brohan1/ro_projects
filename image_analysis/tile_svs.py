# This script takes in a raw .SVS image and outputs a directory of tiled .tiff images
# Rohan Borkar 2015
import math, os, time, shutil
from PIL import Image
from sys import argv
import openslide as op
from openslide import deepzoom

script, svsimage = argv

# How big your tiles will be. Must be square tiles
tiledim = 1000

# Saves .tiff images to this directory: Tiled_[imagename]
tiledir = 'Tiled_'

def tile(file1):
	# Check if directory exists. If it does, delete the old one and make a new one.
	if os.path.isdir(tiledir + svsimage) == True:
		shutil.rmtree(tiledir + svsimage)
	else:
		pass
	os.mkdir(tiledir + svsimage)
	time0 = time.time()

	# Here starts the actual tiling code
	# Opens slide object as 'img'. Not sure how this is different from op.OpenSlide(file1)
	img = op.open_slide(file1)

	# Open slide in "deepzoom" for tiling
	deep = op.deepzoom.DeepZoomGenerator(img, tile_size=tiledim, overlap=0, limit_bounds=False)

	# Record which "level" in deepzoom the image is in (biggest image), and calculate how many tiles to save
	level_count = deep.level_tiles.index(max(deep.level_tiles))
	tile_x, tile_y = max(deep.level_tiles)

	image_count = 1
	kept_tiles = 0
	for a in range( 0, tile_y-1):
		for b in range( 0, tile_x-1):
			# Extract tile from deepzoom image
			cropped_image = deep.get_tile( (level_count), (b, a) )

			# Now analyze the tile: convert to gray and extract data
			gim = cropped_image.convert( 'L' )
			pixels = list(gim.getdata())
			# Find the average pixel intensity
			level = sum(pixels) / len(pixels)

			# If average intensities are too close to "black" or "white", omit the deepzoom tile
			if level >= 230 or level <= 25:
				pass
			# Otherwise save the crop
			else:
				cropped_image.save( './' + tiledir + svsimage + '/' + str( image_count ) + '.tiff', 'TIFF' )
				kept_tiles += 1
			image_count += 1

	print "Saved %r tiles, omitted %r tiles. Process took %r seconds." % (kept_tiles, image_count - kept_tiles, round((time.time() - time0), 2))


tile(svsimage)

myimg.svs
