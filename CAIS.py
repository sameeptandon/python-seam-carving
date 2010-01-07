"""
CAIS.py
An implementation of the Seam Carving / Content Aware Imaging Scaling algorithm.

Dependencies: 
Scipy, Numpy, and Python Imaging Library

Summary of methods:
print_fn(s) prints s to the terminal only when the verbose option is selected
grayscale_filter(img) returns a grayscale version of an Image object img
slow_gradient_filter(img) returns the Sobel Operator applied to img (slow implementation)
gradient_filter(img) returns the Sobel Operator applied to img (fast implementation)
img_transpose(img) returns the transpose of an Image object img 
find_horizontal_seam(img) finds the lowest energy horizontal path in a grayscale image 
find_vertical_seam(img) finds the lowest energy vertical path in a grayscale image
mark_seam(img, path) marks all the pixels in path on img in white
delete_vertical_seam(img, path) removes all the pixels in a vertical path from img
delete_horizontal_seam(img, path) removes all the pixels in a horizontal path from img
add_vertical_seam(img, path) adds the average of the pixels near vertical path to img
add_horizontal_seam(img, path) adds the average of the pixels near horizontal path to img
vector_avg(u,v) returns the component average of two vectors u, v
CAIS(input_img, resolution, output_img, mark) is the controller method 

Copyright (c) 2010, Sameep Tandon
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Sameep Tandon nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Sameep Tandon BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import Image
from scipy.ndimage.filters import generic_gradient_magnitude, sobel
import numpy
from math import fabs
import sys


inf = 1e1000
verbose = False 

def print_fn( s ):
	
	"""
	prints diagnostic messages if the verbose option has been enabled
	@s: string s to print
	"""
	
	global verbose
	if verbose:
		print s

def grayscale_filter ( img ):
	
	""" 
	Takes an image and returns a grayscale image using floats
	@img: the input img
	"""
	return img.convert("F")
	
def slow_gradient_filter ( img ):
	
	""" 
	Takes a grayscale img and returns the magnitude of the gradient operator on the image. Implements the Sobel operator. 
	See http://en.wikipedia.org/wiki/Sobel_operator for details on the Sobel operator 
	@img: a grayscale image represented in floats
	"""
	gradient = Image.new("F", img.size, 0.0)
	max_x, max_y = img.size
	input = img.load( )
	output = gradient.load( )
	for y in range(1, max_y-1):
		for x in range(1, max_x-1):
			dx_pos = 4 * input[x,y] - 2 * input[x-1,y] - input[x-1,y+1] - input[x-1,y-1]
			dx_neg = 4 * input[x,y] - 2 * input[x+1,y] - input[x+1,y+1] - input[x+1,y-1]
			dx = dx_pos - dx_neg
		
			dy_pos = 4 * input[x,y] - 2 * input[x,y-1] - input[x+1,y-1] - input[x-1,y-1]
			dy_neg = 4 * input[x,y] - 2 * input[x,y+1] - input[x+1,y+1] - input[x-1,y+1]
			dy = dy_pos - dy_neg
			
			output[x,y] = fabs(dx) + fabs(dy)
			
	return gradient
	
def gradient_filter ( im ):
	
	"""
	Takes a grayscale img and retuns the Sobel operator on the image. Fast thanks to Scipy/Numpy. See slow_gradient_filter for
	an implementation of what the Sobel operator is doing
	@im: a grayscale image represented in floats
	"""
	print_fn("Computing energy function using the Sobel operator")
	im_width, im_height = im.size 
	im_arr = numpy.reshape( im.getdata( ), (im_height, im_width) )
	sobel_arr = generic_gradient_magnitude( im, derivative=sobel )
	gradient = Image.new("F", im.size)
	gradient.putdata( list( sobel_arr.flat ) )
	
	return gradient
	
def img_transpose(im):
	
	""" 
	Returns the transpose of the Image object
	@img: input image
	"""
	
	im_width, im_height = im.size
	cost = numpy.zeros( im.size )	
	im_arr = numpy.reshape( im.getdata( ), (im_height, im_width) )
	im_arr = numpy.transpose(im_arr)
	im = Image.new(im.mode, (im_height, im_width) )	
	im.putdata( list( im_arr.flat) )
	return im
	
	
def find_horizontal_seam ( im ):
	
	"""
	Takes a grayscale img and returns the lowest energy horizontal seam as a list of pixels (2-tuples). 
	This implements the dynamic programming seam-find algorithm. For an m*n picture, this algorithm
	takes O(m*n) time 
	@im: a grayscale image
	"""
	
	im_width, im_height = im.size
	
	cost = numpy.zeros( im.size )
	
	im_arr = numpy.reshape( im.getdata( ), (im_height, im_width) )
	im_arr = numpy.transpose(im_arr)
	for y in range(im_height):
		cost[0,y] = im_arr[0, y]
	
	print_fn( "Starting Seam Calculations..." )
		
	for x in range(1, im_width):
		if x % 200 == 0:
			print_fn( x )
		for y in range(im_height):
			if y == 0:
				min_val = min( cost[x-1,y], cost[x-1,y+1] )
			elif y < im_height - 2:
				min_val = min( cost[x-1,y], cost[x-1,y+1] )
				min_val = min( min_val, cost[x-1,y-1] )
			else:
				min_val = min( cost[x-1,y], cost[x-1,y-1] )
			cost[x,y] = im_arr[x,y] + min_val
			
	print_fn( "Reconstructing Seam Path..." )
	min_val = inf
	path = [ ]
	
	for y in range(im_height):
		if cost[im_width-1,y] < min_val:
			min_val = cost[im_width-1,y]
			min_ptr = y 
			
	pos = (im_width-1,min_ptr)
	path.append(pos) 
	
	while pos[0] != 0:
		val = cost[pos] - im_arr[pos]
		x,y = pos
		if y == 0:
			if val == cost[x-1,y+1]:
				pos = (x-1,y+1) 
			else:
				pos = (x-1,y)
		elif y < im_height - 2:
			if val == cost[x-1,y+1]:
				pos = (x-1,y+1) 
			elif val == cost[x-1,y]:
				pos = (x-1,y)
			else:
				pos = (x-1,y-1)
		else:
			if val == cost[x-1,y]:
				pos = (x-1,y)
			else:
				pos = (x-1,y-1) 
		
		path.append(pos)
 
	
	print_fn( "Reconstruction Complete." )
	return path
	
def find_vertical_seam ( im ): 
	
	"""
	Takes a grayscale img and returns the lowest energy vertical seam as a list of pixels (2-tuples). 
	This implements the dynamic programming seam-find algorithm. For an m*n picture, this algorithm
	takes O(m*n) time 
	@im: a grayscale image
	"""
	
	im = img_transpose(im)
	u = find_horizontal_seam(im)
	for i in range(len(u)):
		temp = list(u[i])
		temp.reverse()
		u[i] = tuple(temp)
	return u
	
def mark_seam (img, path):
	
	"""
	Marks a seam for easy visual checking
	@img: an input img
	@path: the seam
	"""
	pix = img.load()
	
	print_fn( "Marking seam..." )
	if img.mode == "RGB": 
		for pixel in path:
			pix[pixel] = (255,255,255)
	else:
		for pixel in path:
			pix[pixel] = 255
	
	print_fn( "Marking Complete." )		
	return img
			

def delete_horizontal_seam (img, path):
	
	"""
	Deletes the pixels in a horizontal path from img
	@img: an input img
	@path: pixels to delete in a horizontal path
	"""
	print_fn( "Deleting Horizontal Seam..." )
	#raise Exception
	img_width, img_height = img.size
	i = Image.new(img.mode, (img_width, img_height-1))
	input  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for y in range(img_height):
		for x in range(img_width):
			if (x,y) not in path_set and x not in seen_set:
				output[x,y] = input[x,y]
			elif (x,y) in path_set:
				seen_set.add(x)
			else:
				output[x,y-1] = input[x,y]
	
	print_fn( "Deletion Complete." )			
	return i
			

def delete_vertical_seam (img, path):
		
	"""
	Deletes the pixels in a vertical path from img
	@img: an input img
	@path: pixels to delete in a vertical path
	"""
	print_fn( "Deleting Vertical Seam..." )
	#raise Exception
	img_width, img_height = img.size
	i = Image.new(img.mode, (img_width-1, img_height))
	input  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for x in range(img_width):
		for y in range(img_height):
			if (x,y) not in path_set and y not in seen_set:
				output[x,y] = input[x,y]
			elif (x,y) in path_set:
				seen_set.add(y)
			else:
				output[x-1,y] = input[x,y]
	
	print_fn( "Deletion Complete." )		
	return i
	
def add_vertical_seam(img, path):
	"""
	Adds the pixels in a vertical path from img
	@img: an input img
	@path: pixels to delete in a vertical path
	"""

	print_fn( "Adding Vertical Seam..." )
	img_width, img_height = img.size
	i = Image.new(img.mode, (img_width + 1, img_height) )
	input  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for x in range(img_width):
		for y in range(img_height):
			if (x,y) not in path_set and y not in seen_set:
				output[x,y] = input[x,y]
			elif (x,y) in path_set and y not in seen_set:
				output[x,y] = input[x,y]
				seen_set.add( y )
				if x < img_width -2:
					output[x+1,y] = vector_avg(input[x,y], input[x+1,y])
				else:
					output[x+1,y] = vector_avg(input[x,y], input[x-1,y])
			else:
				output[x+1,y] = input[x,y]
				
	print_fn( "Addition Complete." )
	return i
	
def add_horizontal_seam(img, path):
	
	"""
	Adds the pixels in a horizontal path from img
	@img: an input img
	@path: pixels to delete in a horizontal path
	"""
	print_fn( "Adding Horizontal Seam..." )
	img_width, img_height = img.size
	i = Image.new(img.mode, (img_width, img_height+1) )
	input  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for y in range(img_height):
		for x in range(img_width):
			if (x,y) not in path_set and x not in seen_set:
				output[x,y] = input[x,y]
			elif (x,y) in path_set and x not in seen_set:
				output[x,y] = input[x,y]
				seen_set.add( x )
				if y < img_height -2:
					output[x,y+1] = vector_avg(input[x,y], input[x,y+1])
				else:
					output[x,y+1] = vector_avg(input[x,y], input[x,y-1])
			else:
				output[x,y+1] = input[x,y]
				
	print_fn( "Addition Complete." )
	return i		
				
def vector_avg (u, v):
	
	"""
	Returns the component average between each vector
	@u: input vector u
	@v: input vector v
	"""
	w = list(u)
	for i in range(len(u)):
		w[i] = (u[i] + v[i]) / 2
	return tuple(w)
	
	
def argmin(sequence, vals):

	"""
	Returns the argmin of sequence, where vals is the mapping of the sequence
	@sequence: a list of vars that map to vals
	@vals: the vals of the sequence
	example: argmin( ('x','y','z'), [2,3,1] ) returns 'z'
	"""

	return sequence[ vals.index(min(vals)) ]
	
def CAIS(input_img, resolution, output, mark):
	
	"""
	The main controller method
	@input_img: the file name of the input_img
	@resolution: the resolution to resize the image to
	@output: the file name of the output_img
	@mark: Useful debugging feature to show which seams are being deleted 
	"""
	
	input = Image.open(input_img)
	im_width, im_height = input.size
	marked = [ ]

	while im_width > resolution[0]:
		u = find_vertical_seam(gradient_filter(grayscale_filter(input)))
		if mark:
			marked.append(u)
		input = delete_vertical_seam(input, u)
		im_width = input.size[0]
		
	while im_width < resolution[0]:
		u = find_vertical_seam(gradient_filter(grayscale_filter(input)))
		input = add_vertical_seam(input, u)
		im_width = input.size[0]
				
	while im_height > resolution[1]:
		v = find_horizontal_seam(gradient_filter(grayscale_filter(input)))
		if mark:
			marked.append(u)
		input = delete_horizontal_seam(input,v)
		im_height = input.size[1]
	while im_height < resolution[1]:
		v = find_horizontal_seam(gradient_filter(grayscale_filter(input)))
		input = add_horizontal_seam(input,v)
		im_height = input.size[1]
		
	input.save(output, "JPEG")
	
	if mark and marked != [ ]:
		mark_seam(Image.open(input_img), marked).show( )
	

def main():
	from optparse import OptionParser
	import os 
	usage = "usage: %prog -i [input image] -r [width] [height] -o [output name] \n" 
	usage+= "where [width] and [height] are the resolution of the new image"
	parser = OptionParser(usage=usage)
	parser.add_option("-i", "--image", dest="input_image", help="Input Image File")
	parser.add_option("-r", "--resolution", dest="resolution", help="Output Image size [width], [height]", nargs=2)
	parser.add_option("-o", "--output", dest="output", help="Output Image File Name")
	parser.add_option("-v", "--verbose", dest="verbose", help="Trigger Verbose Printing", action="store_true")
	parser.add_option("-m", "--mark", dest="mark", help="Mark Seams Targeted. Only works for deleting", action="store_true")
	(options, args) = parser.parse_args()
	if not options.input_image or not options.resolution:
		print "Incorrect Usage; please see python CAIS.py --help"
		sys.exit(2)
	if options.verbose:
		global verbose
		verbose = True
	if not options.output:
		output = os.path.splitext(options.input_image)[0] + ".CAIS.jpg"
	else:
		output = options.output
	if options.mark:
		mark = True
	else:
		mark = False
	try: 
		input_image = options.input_image
		resolution = ( int(options.resolution[0]), int(options.resolution[1]) )
	except:
		print "Incorrect Usage; please see python CAIS.py --help"
		sys.exit(2)
		
	CAIS(input_image, resolution, output, mark)
	

if __name__ == "__main__":
	main()
	