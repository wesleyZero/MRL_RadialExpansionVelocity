# Modules________________________________________________________________________


# import collections
# from concurrent.futures import process
# from email.errors import CloseBoundaryNotFoundDefect, FirstHeaderLineIsContinuationDefect
# from ipaddress import collapse_addresses
# import math
# from pickletools import uint1
# from re import I
# from this import d
# from argparse import ONE_OR_MORE
# Modules I actually Added___________________________________________
# from tkinter import W
# from tkinter import W
# from sre_constants import OP_IGNORE
# from msilib.schema import Directory
# from xml.sax import default_parser_list
from cProfile import label
import numpy as np
import cv2 
import tifffile
from scipy import fftpack
from matplotlib import pyplot as plt
# Experimental Modules_______________________________________________
import os
import copy
import glob  # get list of filenames from a directory
import csv
import re  # regular expressions
import tracemalloc
import time
import sys
import math
#Memory Traceing________________________________________________________________
# tracemalloc.start()

# Debugging Output_______________________________________________________________
DEBUG_arr2avi = False
DEBUG_arr2avi_color = False
DEBUG_detectCellLocations = False
seperator = "______________________________________________________________\n"

# Class for MRL Image Analysis___________________________________________________

class MRLcv:
	"""
	This class is for the UCSB Materials Research Laboratory, Pitenis Lab,
	for the purpose of using computer vision (AKA image analysis) for 
	scientific research.
	"""


	# IMAGE | PRESENTATION 			_________________________________________________

	@staticmethod
	def addAnnotation2img(image: np.uint8, txt: str, x: float, y: float, textColor=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_PLAIN,
						  fontScale=0.5,
						  thickness=1, lineType=cv2.LINE_AA):
		hasColor = MRLcv.isAColorImage(image)
		if hasColor:
			(width, height, color) = image.shape
			location = int(width * x), int(height * y)
		else:
			(width, height) = image.shape
			location = int(width * x), int(height * y)
		annotatedImage = cv2.putText(img=image, text=txt, org=location,
									 fontFace=fontFace,
									 fontScale=fontScale, color=textColor,
									 thickness=thickness, lineType=lineType)
		return annotatedImage

	@staticmethod
	def concatImg(arr1: np.uint8, arr2: np.uint8, horizontalConcat: bool = True):
		"""
		Concatenates the two images. Returns the resulting image.

		@param arrX: 
				This array must be in the format of [x, y, color(optional)]
		"""
		# Check if the shapes are the same, if not print an error
		if arr1.shape != arr2.shape:
			print("MRLcv.concatImgH ERROR")
			return

		hasColor = True if (arr1.ndim == 3) else False

		concatAxis = 1 if horizontalConcat else 0

		if hasColor == True:
			concatFrame = np.append(
				arr1[:, :, :], arr2[:, :, :], axis=concatAxis)
		else:
			concatFrame = np.append(arr1[:, :], arr2[:, :], axis=concatAxis)
		return concatFrame

	# IMAGE | FORMAT				_________________________________________________

	@staticmethod
	def bw2colorImg(arr: np.uint8):
		x = arr.shape[0]
		y = arr.shape[1]
		numColorChannels = 3
		colorArr = np.ndarray(shape=(x, y, numColorChannels), dtype=np.uint8)

		# Each color channel has equal amplitude
		for colorChannel in range(0, numColorChannels):
			colorArr[:, :, colorChannel] = arr
		return colorArr

	@staticmethod
	def color2bwImg(image: np.uint8):
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# IMAGE | HELPER				_________________________________________________

	@staticmethod
	def getImageWidth(image: np.uint8):
		if MRLcv.isAColorImage(image):
			width, height, colorChannel = image.shape
			return width
		else:
			width, height = image.shape
		return width

	@staticmethod
	def getImageHeight(image: np.uint8):
		if MRLcv.isAColorImage(image):
			width, height, colorChannel = image.shape
			return height
		else:
			width, height, = image.shape
		return height

	@staticmethod
	def getFrameShape(image: np.uint8):
		width = MRLcv.getImageWidth(image)
		height = MRLcv.getImageHeight(image)
		color = 3 if MRLcv.isAColorImage(image) else None
		if color:
			shape = (width, height, color)
		else:
			shape = (width, height)
		return shape

	@staticmethod
	def isAColorImage(arr: np.uint8):
		"""
		Determines if an image given (as an array) has color channels or 
		is grayscale. 

		@param arr:
				This is the image file. Has the format of [x, y] or [x, y, RBG]
		"""
		if arr.ndim == 3:
			return True
		elif arr.ndim == 2:
			return False
		else:
			print("MRLcv.isAColorImage ERROR the image provided is neither" +
				  " color nor a B&W image. arr.shape = " + str(arr.shape))
			return

	@staticmethod
	def showImage(arr: np.array, winName="test", time: int = 2000):
		"""
		Uses openCV to display the image given in the array.

		@param: arr: np.array
				The array that contains the image data.
		@return:
				returns nothing
		"""
		dim = arr.ndim
		if dim < 2 or dim > 3:
			print("MRLcv.showImage ERROR the dimensions of the image are not correct")
			return
		cv2.imshow(winName, arr)
		cv2.waitKey(time)

	# IMAGE | PROCESSING			_________________________________________________

	#I THINK I NEED TO USE SINS AND COSINE MASKING MATRICIES TO FIX THE RIPPLING
	#EFFECT THAT OCCURS IN THE FOURIER BAND PASS FILTER
	@staticmethod
	def addFFTfilter2img(img, freqMin: float = 0.0, freqMax=1.0):
		"""MUST BE BLACK AND WHITE IMAGE only acts as a high pass filter"""
		f = np.fft.fft2(img)
		fshift = np.fft.fftshift(f)
		rows, cols = img.shape
		# print(fshift)

		if freqMax <= freqMin:
			print('error: image will be zero addFFTfilter2img')

		middle_row, middle_col = int(rows/2), int(cols/2)
		#High Pass Filter: 
		if freqMin > 0:
			r_min = middle_row - int(freqMin * middle_row)
			r_max = middle_row + int(freqMin * middle_row)
			c_min = middle_col - int(freqMin * middle_col)
			c_max = middle_col + int(freqMin * middle_col)
			fshift[r_min: r_max, c_min: c_max] = 0
		
		#Low Pass Filter | get row and col indicies to zero out 
		if freqMax < 1.0:
			# r_min = int(middle_row * freqMax) 
			# r_max = rows - int(rows/2 * freqMax)
			# c_min = int(cols/2 * freqMax)
			# c_max = rows - int(cols/2 * freqMax)

			freqMax = 1.0 - freqMax
			r_min = int(freqMax * middle_row)
			r_max = rows - int(freqMax * middle_row)
			c_min = int(freqMax * middle_col)
			c_max = cols - int(freqMax * middle_col)		

			#THIS CODE IS NOT OPTIMIZED FOR SPEED
			# Low Pass Filter | get row and col indicies to zero out 
			for r in range(rows):
				for c in range(cols):
					if r < r_min or r > r_max or c < c_min or c > c_max:
							fshift[r,c] = 0
					
		
		f_ishift = np.fft.ifftshift(fshift)
		img_HPF = np.fft.ifft2(f_ishift)
		img_HPF = np.abs(img_HPF)
		img_HPF = MRLcv.uint_to_uint8(img_HPF)
		return img_HPF

	@staticmethod
	def addThresholdFilter2img(img: np.uint8, min: int = 0, max: int = 255, thresFilter=cv2.THRESH_TOZERO):
		"""This image filter will convert any image given to a black and white imaged because the threshold 
		filter is only valid for one color channel. CHECK THE TYPE OF THERESHOLDING SPECIFIED BY CV2"""

		img = MRLcv.color2bwImg(img) if MRLcv.isAColorImage(img) else img
		if min < 0 or min > 255:
			return None
		if max < 0 or max > 255:
			return None
		if max < min:
			return None
		dst = img.copy
		th, dst = cv2.threshold(img, min, max, thresFilter)
		dst = MRLcv.bw2colorImg(dst) #EXPERIMENTAL``
		return dst

	@staticmethod
	def addHoughesCircles2img(img: np.uint8):

		img = MRLcv.color2bwImg(img) if MRLcv.isAColorImage(img) else img
		# Blur using 3 * 3 kernel.
		# Blur using 3 * 3 kernel.
		img = cv2.blur(img, (3, 3))
  
		# Apply Hough transform on the blurred image.
		detected_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 5, maxRadius = 512)
		
		# Read image.

		
		# Convert to grayscale.
		# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		

		# gray_blurred = cv2.blur(gray, (3, 3))
		
		# Apply Hough transform on the blurred image.
		detected_circles = cv2.HoughCircles(img, 
						cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
					param2 = 30, minRadius = 1, maxRadius = 40)
		
		# Draw circles that are detected.
		if detected_circles is not None:
		
			# Convert the circle parameters a, b and r to integers.
			detected_circles = np.uint16(np.around(detected_circles))
		
			for pt in detected_circles[0, :]:
				a, b, r = pt[0], pt[1], pt[2]
		
				# Draw the circumference of the circle.
				cv2.circle(img, (a, b), r, (0, 255, 0), 2)
		
				# Draw a small circle (of radius 1) to show the center.
				cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
				cv2.imshow("Detected Circle", img)
				cv2.waitKey(0)
		else:
			print('no circles detected')
		return img
		
	@staticmethod
	def addHoughesCircles2img_COPYPASTE():
		# Read image.
		img = cv2.imread('circles00.jpg', cv2.IMREAD_COLOR)
		
		# Convert to grayscale.
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# Blur using 3 * 3 kernel.
		# gray_blurred = cv2.blur(gray, (3, 3))
		# MRLcv.addGaussianFilter2Video([gray], (7,7))
		gray_blurred = cv2.GaussianBlur(gray, (3,3), 1.5)
		# Apply Hough transform on the blurred image.
		detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1 = 40,param2 = 255, minRadius = 1, maxRadius = 512)
		
		# Draw circles that are detected.
		if detected_circles is not None:
		
			# Convert the circle parameters a, b and r to integers.
			detected_circles = np.uint16(np.around(detected_circles))
		
			for pt in detected_circles[0, :]:
				a, b, r = pt[0], pt[1], pt[2]
		
				# Draw the circumference of the circle.
				cv2.circle(img, (a, b), r, (0, 255, 0), 1)
		
				# Draw a small circle (of radius 1) to show the center.
				cv2.circle(img, (a, b), 1, (0, 0, 255), 1)
				# cv2.imshow("Detected Circle000", img)
				# cv2.waitKey(250)
			cv2.imshow("Detected Circle", img)
			cv2.waitKey(0)
		else:
			print("ERROR NO CIRCLES")
		return img
			

	def addThresholdBandFilter(img:np.uint8, min:int=0, max:int=255, replaceValue:int=0):
		if MRLcv.isAColorImage(img): img = MRLcv.bw2colorImg(img)
		for row in image:
			for pixel in row:
				if pixel >= min and pixel <= max:
					pixel = replaceValue
		

	# VIDEO | PROCESSING			_________________________________________________

	@staticmethod
	def addCannyEdgeFilter(arr: np.uint8, thres1: int = 0, thres2: int = 255, apertureSize:int=3):
		"""
		@param: arr: np.uin8
				This array has to have the format of [t, x, y]
		@return:
				An array of image that have the high pass filter applied to them. 
				Specifically, this would be a canny edge detector. B&W image will 
				have color channels for compatibility with color videos.  
		"""
		hasColor = MRLcv.isAcolorVideo(arr)
		timeRange = arr[:, 0, 0, 0].size if hasColor else arr[:, 0, 0].size
		videoShape = (0, arr.shape[1], arr.shape[2], 3) if hasColor else (
			0, arr.shape[1], arr.shape[2])
		cannyFilteredVideo = np.ndarray(shape=videoShape, dtype=np.uint8)

		for t in range(0, timeRange):
			frame = arr[t, :, :, :] if hasColor else arr[t, :, :]
			cannyFrame = cv2.Canny(
				frame, threshold1=thres1, threshold2=thres2, apertureSize=apertureSize)
			cannyFrame = MRLcv.bw2colorImg(cannyFrame)
			cannyFilteredVideo = np.append(
				cannyFilteredVideo, [cannyFrame], axis=0)
		return cannyFilteredVideo

	@staticmethod
	def addFFT2video(video, freqMin: float = 0.0, freqMax: float = 1.0):
		"""Applies a FFT filter to each GRAYSCALE frame in the video"""
		video = MRLcv.color2bwVideo(video)
		t, x, y = video.shape
		fftVideo = np.ndarray(shape=(0, x, y), dtype=np.uint8)
		for img in video:
			fftImg = MRLcv.addFFTfilter2img(img, freqMin, freqMax)
			fftVideo = np.append(fftVideo, [fftImg], axis=0)
		fftVideo = MRLcv.addColorChannels2bwVideo(fftVideo)  # EXPERIMENTAL
		return fftVideo

	@staticmethod
	def __getVideoProcessorParams(processName2mod: str, modifiedParams, numVideos: int):
		"""
		numVideos and param value list lengths must be the same length

		@param: paramModifications
				This MUST BE A LIST for this function to work.
		@return:
				Returns a parameter Tuple, which can be used to unpack straight
				into a functions argument. 
		"""
		defaultParams = {
			"gaussian": {
				"kernalSize": (3, 3)
				,'sigmaY' : 1.0
				,'sigmaX' : 1.0
			},
			"addAnnotation2video":{
				"txt": "_" 
				,"x": 0.0
				,"y": 0.95
				,"textColor": (0, 255, 0)
				,"fontFace": cv2.FONT_HERSHEY_PLAIN
				,"fontScale": 1.0
			},
			"concat2videoGrid":{
			},
			"cannyEdgeDetector":{
				"thres1": 0
				,"thres2": 255
				,'aperatureSize': 3
			},
			"addKeypoints2video":{
				"blobDetectorParams": MRLcv.getBlobParams()
				,"keypointColor": (0, 255, 0)
			},
			"addFFT2video":{
				"freqMin": 0.0
				,"freqMax": 1.0
			},
			'addThreshold2Video':{
				'min': 0 
				,'max': 255
				,'thresFilter' : cv2.THRESH_TOZERO
			},
			'addHoughesCircles':{
				
			}
		}

		if processName2mod in defaultParams:
			params = defaultParams[processName2mod]
			for paramName in params:
				params[paramName] = [params[paramName]
									 for _ in range(numVideos)]

			for modParamName in modifiedParams:
				if modParamName in params:
					params[modParamName] = modifiedParams[modParamName]
				else:
					print("ERROR param name not found: " + modParamName)
			return params
		else:
			print("__getVideoProcessorParams ERROR processName not found: " +
				  processName2mod)
			return

	@staticmethod
	def	getIterativeParams(modVarName:str=None, num:int=0,  modVarMin=0.0, modVarMax=1.0, type=1.0, currentParams=None):
		# annoteParams = {
		# "txt": ["life is poop, poop is life..." for _ in range(1, numVideosPerSeries+1)]
		# ,"fontScale": [1.0 for _ in np.linspace(1.0, 3.0, num=numVideosPerSeries)]
		# ,"textColor": [(0, 255, 0) for i in np.linspace(0, 255, numVideosPerSeries)]
		# ,"x": MRLcv.linSpaceList(0, 0, numVideosPerSeries)
		# ,"y": MRLcv.linSpaceList(.95, .95, numVideosPerSeries)
		# }
		params = {}
		if num > 0:
			params = { modVarName : MRLcv.linSpaceList(modVarMin, modVarMax, num,type) }
		if currentParams is not None:
			currentParams[modVarName] = params[modVarName]
			return currentParams
		return params 

	@staticmethod
	def __getFirstPramsFromDict(paramDict):
		params = []
		# paramDict = copy.deepcopy(paramDict) #delete me if bugs arrise
		for paramName in paramDict:
			params.append(paramDict[paramName].pop(0))
		return tuple(params)

	@staticmethod 
	def videoProcessor(origVideos: list, processName: str, modifiedParams, debugOutput: bool = False):
		"""
		Given a list of videos, go through each video and process the video 
		with a filter provided in the modifiedParams dictionary:

		@param	origVideos
				This is a list of videos to process.
				Requrirements: Must be the same length as params, so that each 
				video has a corresponding process, with parameter conditions.
		@param	processName
				The name of the filter or computer vision video process to 
				perform onto the list of videos. 
		@param	modifiedParams
				This is a list of parameters for the provided processName. The 
				structure of the dictionary must be as so...
				ex: for processName='gaussian' modifiedParams={'paramName':paramValue}	
		"""
		processedVideos = []
		numVideos = len(origVideos)
		if 'blobDetectorParams' not in modifiedParams:
			# DELETE ME IF THERE ARE BUGS
			modifiedParams = copy.deepcopy(modifiedParams)
		paramDict = MRLcv.__getVideoProcessorParams(processName, modifiedParams, numVideos)
		videoIndex = 0
		for video in origVideos:
			
			params = MRLcv.__getFirstPramsFromDict(paramDict)
			if debugOutput == False:
				print("#" + str(videoIndex) +
					  " | videoProcessor | processName = " + processName, end="")
				print(" | args | = " + str(params))
				# snap = memoryTrace(snap=snap, label='Video Processor start')
				videoIndex += 1
			if processName == "gaussian":
				video = MRLcv.addGaussianFilter2Video(video, *params)
			elif processName == "addAnnotation2video":
				video = MRLcv.addAnnotation2Video(video, *params)
			elif processName == "concat2videoGrid":
				# Base Case
				if len(origVideos) < 4:
					print("not enough videos to make a grid")
					return origVideos
				video = []
				for _ in range(4):
					video.append(origVideos.pop(0))
				video = tuple(video)
				video = MRLcv.concat2videoGrid(*video)
				# 4x4 grid Recursive Case
				if len(processedVideos) == 3 and len(origVideos) == 0:
					processedVideos.append(video)
					return MRLcv.videoProcessor(processedVideos, processName, modifiedParams)
			elif processName == "cannyEdgeDetector":
				video = MRLcv.addCannyEdgeFilter(video, *params)
			elif processName == "addKeypoints2video":
				# MAKE SURE YOU SET UP THE PARAMS CORRECT
				video = MRLcv.addKeypoints2video(video, *params)
			elif processName == "addFFT2video":
				video = MRLcv.addFFT2video(video, *params)
			elif processName == "getVideoKeypoints":
				# SHOULD I PROCESS KEYPOINTS HERE OR SHOULD I INTEGRATE IT INTO THE ADD KEYPOINT FUNCTIONALITY?
				pass
			elif processName == 'addThreshold2Video':
				video = MRLcv.addThreshold2Video(video, *params)
			elif processName == 'addHoughesCircles':
				video = MRLcv.addHoughesCircles2Video(video, *params)
			else:
				print("\tMRLcv.videoProcessor ERROR processName \'" +
					  str(processName) + "\' not found")
				return origVideos
			processedVideos.append(video)
		return processedVideos

	@staticmethod
	def addGaussianFilter2Video(BWvideo: np.uint8, kernalSize=(3, 3),sigmaX=1.5, sigmaY=1.5):
		"""
		Applies a gaussian blurring filter to each frame in a series of images 
		in the provided array. 

		@param arr:
				Recieves an array in the format of [t, x, y]. 

		@return
		"""
		if MRLcv.isAcolorVideo(BWvideo):
			# print("VIDEO IS NOT BW addGaussian2video")
			BWvideo = MRLcv.color2bwVideo(BWvideo)
		
		timeRange = BWvideo.shape[0]
		x = BWvideo.shape[1]
		y = BWvideo.shape[2]
		hasColor = True if (BWvideo.ndim == 4) else False
		numColorChannels = 3

		blurredArrShape = (
			0, x, y, numColorChannels) if hasColor else (0, x, y)
		blurredArr = np.ndarray(dtype=np.uint8, shape=blurredArrShape)

		for t in range(0, timeRange):
			frame = BWvideo[t, :, :, :] if hasColor else BWvideo[t, :, :]
			gaussianBlurFrame = cv2.GaussianBlur(
				frame, ksize=kernalSize, sigmaX=0, sigmaY=0)
			blurredArr = np.append(blurredArr, [gaussianBlurFrame], axis=0)
		return MRLcv.addColorChannels2bwVideo(blurredArr)

	@staticmethod
	def addThreshold2Video(video: np.uint8, min: int = 0, max: int = 255):
		video = MRLcv.color2bwVideo(video) if MRLcv.isAcolorVideo(video) else video

		filterVid = None
		for frame in video:
			thresFrame = MRLcv.addThresholdFilter2img(frame, min, max)
			filterVid = MRLcv.addFrame2video(filterVid, thresFrame)
		# filterVid = MRLcv.addColorChannels2bwVideo(filterVid)
		return filterVid


	@staticmethod
	def annotateParamValue2videos(videos, annoteParams, processParams, varName:str, videos_kp=None):
		annoteParams = MRLcv.getAnnoteParam_withParamValue(annoteParams, processParams, varName)
		# annoteParams = MRLcv.videoProcessorTextAppendage(annoteParams, "_poop1")
		# annoteParams = MRLcv.videoProcessorTextReplace(annoteParams, "life is poop, poop is life")

		videos = MRLcv.videoProcessor(videos, "addAnnotation2video", annoteParams)
		if videos_kp is not None:
			videos_kp = MRLcv.videoProcessor(videos_kp, "addAnnotation2video", annoteParams)
			return videos, videos_kp
		return videos

	@staticmethod
	def setNewDefaultParam(paramName, paramValue, num:int=0, currentParams=None):
		
		params = {}
		if num > 0:
			params = { paramName : [paramValue for _ in range(num) ] }
		if currentParams is not None: 
			currentParams[paramName] = params[paramName]
			return currentParams
		return params 

	def	getIterativeParams(modVarName:str=None, num:int=0,  modVarMin=0.0, modVarMax=1.0, type=1.0, currentParams=None):
		# annoteParams = {
		# "txt": ["life is poop, poop is life..." for _ in range(1, numVideosPerSeries+1)]
		# ,"fontScale": [1.0 for _ in np.linspace(1.0, 3.0, num=numVideosPerSeries)]
		# ,"textColor": [(0, 255, 0) for i in np.linspace(0, 255, numVideosPerSeries)]
		# ,"x": MRLcv.linSpaceList(0, 0, numVideosPerSeries)
		# ,"y": MRLcv.linSpaceList(.95, .95, numVideosPerSeries)
		# }
		params = {}
		if num > 0:
			params = { modVarName : MRLcv.linSpaceList(modVarMin, modVarMax, num,type) }
		if currentParams is not None: 
			currentParams[paramName] = params[paramName]
			return currentParams
		return params 


	# VIDEO | PRESENTATION			_________________________________________________

	@staticmethod
	def concat2videoGrid(topLeftArr: np.uint8, topRightArr: np.uint8, bottomLeftArr, bottomRightArr):
		"""
		concatenates the arrays(videos) to a 2x2 video

		@param X:
				All parameters must be the same dimension. The videowriter object 
				bugs out if you try to make non-square videos apparently?.
		"""
		# Error
		if not (topLeftArr.shape == topRightArr.shape == bottomLeftArr.shape == bottomRightArr.shape):
			print("MRLcv.concatVideo ERROR arrays are not the same shape")
			return

		# Determine the shape of the resulting frames
		shape = list(topLeftArr.shape)
		shape[0] = 0
		shape[1] *= 2
		shape[2] *= 2
		shape = tuple(shape)

		# Video array that will be returned, with the determined frame shape
		videoArr = np.ndarray(shape, dtype=np.uint8)

		# Determine if the video has color or not
		hasColor = True if (topLeftArr.ndim == 4) else False

		# Concatenate each of the 4 frames into a 2x2 grid
		timeRange = topLeftArr[:, 0, 0,
							   0].size if hasColor else topLeftArr[:, 0, 0].size
		for t in range(0, timeRange):
			# Frame Quarters that will be in the 2x2 video
			frameTopLeft = topLeftArr[t, :, :,
									  :] if hasColor else topLeftArr[t, :, :]
			frameTopRight = topRightArr[t, :, :,
										:] if hasColor else topRightArr[t, :, :]
			frameBtmLeft = bottomLeftArr[t, :, :,
										 :] if hasColor else bottomLeftArr[t, :, :]
			frameBtmRight = bottomRightArr[t, :, :,
										   :] if hasColor else bottomRightArr[t, :, :]

			topHalfFrame = MRLcv.concatImg(frameTopLeft, frameTopRight)
			bottomHalfFrame = MRLcv.concatImg(frameBtmLeft, frameBtmRight)
			totalFrame = MRLcv.concatImg(
				topHalfFrame, bottomHalfFrame, horizontalConcat=False)

			videoArr = np.append(videoArr, [totalFrame], axis=0)
		return videoArr

	@staticmethod
	def concatAllVideosTime(allSeriesInTime, videos):
		"""complete me"""
		numColorChannels = 3 if MRLcv.isAcolorVideo(videos[0]) else 1
		if allSeriesInTime is None:
			x = videos[0].shape[1]
			y = videos[0].shape[2]
			allSeriesInTime = np.ndarray(
				shape=(0, x, y, numColorChannels), dtype=np.uint8)
		for vid in videos:
			allSeriesInTime = MRLcv.concatVideoTime(allSeriesInTime, vid)
		return allSeriesInTime

	@staticmethod
	def concatVideoTime(firstVideo: np.uint8, secondVideo: np.uint8):
		"""
		Concatenates the two video with respect to time i.e. returns one video 
		where the second video follows the first video provided.
		"""
		return np.append(firstVideo, secondVideo, axis=0)

	@staticmethod
	def addAnnotation2Video(origVideo: np.uint8, txt: str, x: float, y: float, textColor=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_PLAIN,
							fontScale=0.5,
							thickness=1, lineType=cv2.LINE_AA):
		"The origin of the x , y is top right. They must be normalized between 0 and 1"
		if x < 0 or x > 1 or y < 0 or y > 1:
			print("ERROR x value")
			return
		annotatedVideo = None
		for time in range(0, MRLcv.getVideoTimeRange(origVideo)):
			frame = MRLcv.getFrame(origVideo, time)
			annotatedImage = MRLcv.addAnnotation2img(
				frame, txt, x, y, textColor, fontFace, fontScale, thickness, lineType)
			if annotatedVideo is None:
				frameShape = MRLcv.getFrameShape(annotatedImage)
				annotatedVideo = np.ndarray(
					shape=(0, *frameShape), dtype=np.uint8)
			annotatedVideo = MRLcv.addFrame2video(
				annotatedVideo, annotatedImage)
		return annotatedVideo

	@staticmethod
	def videoProcessorTextAppendage(annoteParams, appendedText: str):
		"""dictionary is first param"""
		annoteParams["txt"] = [
			text + appendedText for text in annoteParams["txt"]]
		return annoteParams

	@staticmethod
	def videoProcessorTextReplace(annoteParams, replacementText: str):
		"""dictionary is first param"""
		annoteParams["txt"] = [replacementText for _ in annoteParams["txt"]]
		return annoteParams

	@staticmethod
	def getAnnoteParam_withParamValue(annoteParams, filterParams, paramName):
		"""Replaces the text with the parameter value"""
		annoteParams["txt"] = [paramName + "= " +
							   str(i) for i in filterParams[paramName]]
		return annoteParams

	@staticmethod
	def addVideoOrganoidTrackerDot(video:np.uint8, label:str, loc):
		""" 
		video has format [t, x, y]
		loc is a tuple and has the format of (x, y) where x, y are arrays(in time) with pixel location"""
		x, y = loc 

		labeledVideo = None	
		for t in range(video[ : , 0, 0, 0].size):
			# make a copy of the original image
			# imageFilledCircle = img.copy()
			imageFilledCircle =  video[t, : , :].copy()

			# define center of the circle

			circle_center = (256,256)
			# define the radius of the circle
			radius = 2
			# draw the filled circle on input image
			circleLoc = (int(float(x[t])),int(float(y[t])))
			cv2.circle(imageFilledCircle, circleLoc, radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
			# display the output image
			# cv2.imshow('Image with Filled Circle',imageFilledCircle)
			# cv2.waitKey(0)
			labeledVideo = MRLcv.addFrame2video(labeledVideo, imageFilledCircle)



		# labeledVideo = video
		return labeledVideo

	# NOT COMPLETE
	@staticmethod
	def appendParamValue(annoteParams, filterParam, paramName):
		"""Replaces the text with the parameter value"""
		annoteParams["txt"] = [paramName + "=" +
							   str(i) for i in filterParam[paramName]]
		return annoteParams

	# VIDEO | FORMAT				_________________________________________________

	@staticmethod
	def color2bwVideo(video):
		video = MRLcv.uint_to_uint8(video)
		if MRLcv.isAcolorVideo(video):
			tMax, x_max, y_max, colorChannels = video.shape
		else:
			tMax, x_max, y_max = video.shape
		bwVideo = np.ndarray(shape=(0, x_max, y_max), dtype=np.uint8)
		for t in range(tMax):
			if MRLcv.isAcolorVideo(video):
				bwImg = cv2.cvtColor(video[t, :, :, :], cv2.COLOR_BGR2GRAY)
			else:
				bwImg = video[t, :, :]
			bwVideo = np.append(bwVideo, [bwImg], axis=0)
		return bwVideo

	@staticmethod
	def addColorChannels2bwVideo(arr: np.uint8):
		# Error
		if arr.ndim != 3:
			return
		# Setup
		numColorChannels = 3
		timeRange = arr.shape[0]
		x = arr.shape[1]
		y = arr.shape[2]
		colorVideo = np.ndarray(
			shape=(0, x, y, numColorChannels), dtype=np.uint8)

		# Process each frame
		for t in range(0, timeRange):
			RBGframe = MRLcv.bw2colorImg(arr[t, :, :])
			colorVideo = np.append(colorVideo, [RBGframe], axis=0)
		return colorVideo

	# VIDEO | DETECTION				_________________________________________________
	@staticmethod
	def getBlobParams(modifiedParams: {str: int} = None):

		# Initalize the parameter object
		blobParams = cv2.SimpleBlobDetector_Params()

		# Set params to (my) default parameter values______
		# Color___________________________________
		# blobParams.filterByColor = False

		# # # Change thresholds_____________________
		blobParams.thresholdStep = 1
		blobParams.minThreshold = 0
		blobParams.maxThreshold = 255
		# blobParams.minRepeatabiltiy = 2
		blobParams.minDistBetweenBlobs = 1

		# # Filter by Area__________________________
		blobParams.filterByArea = True
		radius_min = 0
		radius_max = 173
		minArea  = (4/3) * 3.14159 * (radius_min**2)
		maxArea  = (4/3) * 3.14159 * (radius_max**2)
		blobParams.minArea = minArea
		blobParams.maxArea = maxArea

		# # Filter by Circularity___________________
		# 	#This is a value between 0 and 1
		blobParams.filterByCircularity = True
		blobParams.minCircularity = 0.0
		blobParams.maxCircularity = 1.0

		# # Filter by Convexity______________________
		# 	#This is a value between 0 and 1
		blobParams.filterByConvexity = True
		blobParams.minConvexity = 0.0
		blobParams.maxConvexity = 1.0

		# # Filter by Inertia_________________________
		# 	#This is a value between 0 and 1
		blobParams.filterByInertia = True
		blobParams.minInertiaRatio = 0.0
		blobParams.maxInertiaRatio = 1.0

		# Modifications to the default parameters___________
		# print("dir(params)")
		# print(str(dir(blobParams)))
		if modifiedParams is not None:
			for paramName in modifiedParams:
				# print("modParam =" + str(paramName))
				if paramName in dir(blobParams):
					# print("found : " + str(paramName))
					setattr(blobParams, paramName, modifiedParams[paramName])
				else:
					print("getBlobParams ERROR | parameter name not found=" + paramName)
		# print(str(params))
		return blobParams

	@staticmethod
	def addKeypoints2frame(frame: np.uint8, keypoints, keypointColor):
		"""You Complete Me"""
		frame_with_keypoints = cv2.drawKeypoints(frame, keypoints,
												 np.array([]), keypointColor,
												 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		return frame_with_keypoints

	@staticmethod
	def addKeypoints2video(videoArr: np.uint8, params, keypointColor):
		"""You complete me

		@param videoArr: 
				video array must be in the format of [t, x, y]
		@return 
				returns a color video in the format of [t, x, y, color]
		"""
		hasColor = True if (videoArr.ndim == 4) else False

		# Make a video, iterating though the first index of the array
		timeRange = videoArr[:, 0, 0,
							 0].size if hasColor else videoArr[:, 0, 0].size
		# This is the array that will contain each frame WITH keypoints added
		keypointVidArr = np.ndarray(shape=(0, 512, 512, 3), dtype=np.uint8)

		for time in range(0, timeRange):
			frame = videoArr[time, :, :,
							 :] if hasColor else videoArr[time, :, :]
			frameKeypoints = MRLcv.getFrameKeypoints(frame, params)
			frame_with_keypoints = MRLcv.addKeypoints2frame(frame,
															frameKeypoints, keypointColor)
			keypointVidArr = np.append(keypointVidArr, [frame_with_keypoints],
									   axis=0)
		return keypointVidArr

	@staticmethod
	def keypoints2arr(keypoints, t: int = -1):
		"""
				@param: keypoints
						This is a list of keypoint objects, each object has multiple
						keypoints for a particular frame
				@return allFrameKeypoints
						This is an array of the format [x, y, diameter] or [x, y, d, t] 
						if the time parameter is used.
		"""
		if t == -1:
			allFrameKeypoints = np.ndarray(shape=(0, 3), dtype=np.float64)
		elif t >= 0:
			allFrameKeypoints = np.ndarray(shape=(0, 4), dtype=np.float64)
		else:
			print("keypoints2arr ERROR t parameter is not valid")

		for keyP in keypoints:
			(x, y) = keyP.pt
			diameter = keyP.size
			if t == -1:
				keypointArr = np.uint8([x, y, diameter])
			else:
				keypointArr = np.uint8([x, y, diameter, t])
			allFrameKeypoints = np.append(
				allFrameKeypoints, [keypointArr], axis=0)
		return allFrameKeypoints

	@staticmethod
	def getVideoKeypoints(arr: np.uint8, params: cv2.SimpleBlobDetector):
		"""
		You complete me

		@param arr:np.uint8
				Array must be in the order of [t, x, y]
		"""
		if arr.ndim != 3:
			"MRLcv.getVideoKeypoints ERROR wrong dimensions"
			return
		timeRange = arr[:, 0, 0].size
		detector = cv2.SimpleBlobDetector_create(params)
		vidKeypointsArr = np.ndarray(shape=(0, 4), dtype=np.float64)

		for t in range(0, timeRange):
			frame = arr[t, :, :]
			frameKeypointObjs = detector.detect(frame)
			fkpa = MRLcv.keypoints2arr(frameKeypointObjs, t)
			vidKeypointsArr = np.append(vidKeypointsArr, fkpa, axis=0)
		return vidKeypointsArr

	@staticmethod
	def getFrameKeypoints(frame: np.uint8, params: cv2.SimpleBlobDetector_Params):
		"""You complete me"""
		detector = cv2.SimpleBlobDetector_create(params)
		keypoints = detector.detect(frame)
		return keypoints

	@staticmethod
	def	getKeypointParams(numVideosPerSeries):
		blobDetParamArr = []
		blobAnnoteDict = {}
		blobAnnoteArr = []
		blobParamName = "minThreshold"
		for i in np.linspace(0, 0, numVideosPerSeries):
			# blobDetParamOverrides = {blobParamName: i}
			blobDetParamOverrides = {}
			blobAnnoteArr.append(i)
			blobDetParamArr.append(MRLcv.getBlobParams(blobDetParamOverrides))
		blobAnnoteDict[blobParamName] = blobAnnoteArr

		keypointParams = { 
			"keypointColor": [(0, 255, 0) for _ in np.linspace(0, 255, numVideosPerSeries)],
			"blobDetectorParams": blobDetParamArr}
		return keypointParams

	# VIDEO | Z-TRACKING 			________________________________________________


	# VIDEO | HELPER 				_________________________________________________

	@staticmethod
	def getVideoTimeRange(video: np.int8):
		if MRLcv.isAcolorVideo(video):
			(timeRange, x, y, c) = video.shape
		else:
			(timeRange, x, y) = video.shape
		return timeRange

	@staticmethod
	def isAcolorVideo(arr: np.uint8):
		"""
		Determines if an video given (as an array) has color channels or 
		is grayscale. 

		@param arr:
				This is the image file. Has the format of [t, x, y] or [t, x, y, RBG]
		"""
		if arr.ndim == 4:
			return True
		elif arr.ndim == 3:
			return False
		else:
			print("MRLcv.isAColorVideo ERROR the video provided is neither" +
				  " color nor a B&W video. arr.shape = " + str(arr.shape))
			return

	@staticmethod
	def addFrame2video(video: np.uint8, frame):
		if video is None:
			shape = frame.shape
			video = np.ndarray(shape=(0, *shape),
							   dtype=np.uint8)  # EXPERIMENTAL
		if MRLcv.isAcolorVideo(video):
			frame = frame if MRLcv.isAColorImage(
				frame) else MRLcv.bw2colorImg(frame)
			return np.append(video, [frame], axis=0)
		else:
			frame = MRLcv.color2bwImg(
				frame) if MRLcv.isAColorImage(frame) else frame
			return np.append(video, [frame], axis=0)


	@staticmethod
	def getFrame(video: np.uint8, time: int):
		"""Gets a frame from a video"""
		if MRLcv.isAcolorVideo(video):
			return video[time, :, :, :]
		else:
			return video[time, :, :]

	@staticmethod
	def openVideo(videoFilename: str, dir: str = '/.') -> np.uint8:
		videoCap = cv2.VideoCapture(dir + videoFilename)
		# while frame = videoCap.read(): MRLcv.showImg(frame)
		video = None
		while videoCap.isOpened():
			ret, frame = videoCap.read()
			if ret:
				# MRLcv.showImage(frame)
				video = MRLcv.addFrame2video(video, frame)
			else:
				break
		return video

	# VIDEO | I/O					_________________________________________________

	@staticmethod
	def saveAllVideos(videos, filename, directory: str = "", FPS: int = 5):
		# directory creation
		if not os.path.exists(directory):
			os.makedirs(directory)
		#CREATE AN ERROR IF THE VIDEO FILE ALREADY EXISTS
		
		i = 0
		videos = np.asarray(videos)
		# Multiple Videos
		if videos.ndim == 5:
			for vid in videos:
				if i > 0:
					completeFilename = directory + \
						filename + "_" + str(i) + ".avi"
				else:
					completeFilename = directory + filename + ".avi"
				MRLcv.__arr2avi(vid, completeFilename, fps=FPS)
				i += 1
		# One Color Video
		elif videos.ndim == 4:
			completeFilename = directory + filename + ".avi"
			MRLcv.__arr2avi(videos, completeFilename, fps=FPS)
		# One B&W Video
		elif videos.ndim == 3:
			completeFilename = directory + filename + ".avi"
			MRLcv.__arr2avi(videos, completeFilename, fps=FPS)
		else:
			print("saveAllVideos ERROR ndim")
		return True

	@staticmethod
	def __arr2avi(arr: np.uint8, outFileName="test.avi", fps=1):
		"""
		Generates an *.avi file from a array data structure. 

		@Param arr: np.uint8
				This array has to have dimensions in this order! [t, x, y]
		@param: outFilename
				This is the name of the video file that you want to save. The video
				must have the extension of *.avi but other container extensions 
				*may* be valid (untested).
		@param: hasColor:bool
				If the video is in color (last 3 indicies are BGR) then this 
				parameter should be set to True. B&W videos should have False.
		@param: fps:int
				This is the frames per second of the video file
		@return:
				Returns True/False depending on success of video writing. Writes an 
				*.avi file to the same directory as this python file.
		"""
		# Error
		if arr is None:
			print("MRLcv.arr2avi ERROR the array passed is None")
			return False

		# If the video is color, just use the wrapper function for color & return
		hasColor = True if (arr.ndim == 4) else False

		# Indicies of the Respective Information
		xIndex = 1
		yIndex = 2
		# , 3) if hasColor else (arr.shape[xIndex], arr.shape[yIndex])
		imgSize = (arr.shape[xIndex], arr.shape[yIndex])

		# Make the Video Writing Object, so a video can be made.
		vidWriter = cv2.VideoWriter(outFileName,
									cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
									fps, imgSize, isColor=hasColor)

		# Take each frame, add it to the videoWriting obj, write the file.
		timeRange = arr[:, 0, 0, 0].size if hasColor else arr[:, 0, 0].size
		for time in range(0, timeRange):
			frame = arr[time, :, :, :] if hasColor else arr[time, :, :]
			vidWriter.write(frame)
		print("arr2avi | saved file | \'" + str(outFileName) + "\'")
		vidWriter.release()
		return True

	# DATA | 		 				_________________________________________________

	@staticmethod
	def readTifFile(filename: str) -> np.array:
		"""
		Reads a *.tif file with the provided filename. Returns a numpy array 
		with the data that is in the *.tif file. 

		@param: filename: str
				This is the name of the *.tif that you want to open.
		@return:
				Returns a numpy array, from the data stored in the *.tif file.
		"""
		arr = tifffile.imread(filename)
		if arr is not None:
			return MRLcv.uint_to_uint8(arr)
		else:
			print("MRLcv.readTifFile ERROR file didn't open successfully")
			return None

	# Should I make a version of this function that DOESN"T normalize the values
	@staticmethod
	def uint_to_uint8(uint16arr: np.array) -> np.uint8:
		"""
		Converts an int numpy array to an 8 bit array (unsigned int). 

		WARNING: The values are normalized e.g. the largest value in the 16 bit
		array, may be 28.5% of the max value(2^16 - 1). However that will be 
		normalized, such that the max value in the 16 bit array will be equal 
		to 255 (i.e. 100% of the max value of an 8bit array that is returned)

		@param: uint16arr
				This is the 16 bit unsigned array that you want to convert to 8bit.
		@return:
				Returns an 8 bit unsigned numpy array (normalized).
		"""
		# ERROR HANDLING
		if uint16arr is None:
			print("uint_to_uint8 Error: the 16 bit array given for conversion" +
				  "was None!")
			return None
		# Use only the first 8 bits (of 16) to represent the data
		uint16arr = (uint16arr / uint16arr.max()) * 255
		# truncate the extra 8 bits AKA change the data type
		uint8arr = np.uint8(uint16arr)
		return uint8arr

	@staticmethod
	def getCSVcolData(colLabel: str, csvFilename: str, dir: str = None) -> [str]:
		csvFile = open(csvFilename)
		csvData = []

		csvFileRows = iter(csvFile)
		row = next(csvFileRows)
		row = row.strip('\n')
		row = row.split(',')  # CSV string -> List[ str]

		# row = [varName.strip('\n') for varName in row]

		# Get the index of the column label specified
		try:
			dataIndex = row.index(colLabel)
		except ValueError as err:
			print(err)
			return None

		# Go through the rest of the rows in CSV, extract data
		for row in csvFileRows:
			# row.strip('\n')
			row = row.strip('\n')
			row = row.split(',')
			# print(row[dataIndex])
			csvData.append(row[dataIndex])
		return csvData

	@staticmethod
	def getAllSeriesDataset(varNames: [str], inputCSVdir: str = "./"):
		""" THIS WILL ONLY WORK IF THE ORGANOID LABELS ARE BETWEEN A AND H. LOOK AT
		HOW I IMPLEMENTED THE WAY LABELS FOR NUMBERS AND ORGANOID_LABELS ARE SETUP"""
		# Reading the CSV measurement filenames into a list
		csvFilenameList = glob.glob(inputCSVdir + "*.csv")

		# The new CSV file to be created
		# f = open(outputCSVDir + outputCSVfilename, 'w')
		# writer = csv.writer(f)

		# Make the first row labels for each column
		fileColLabels = ['seriesIndex'] + ['organoidLabel'] + varNames
		# writer.writerow(fileColLabels)
		allOrganoidData = []
		allOrganoidData.append(fileColLabels)

		# Process each CSV file with organoid data in it
		for csvFilename in csvFilenameList:
			# print(csvFilename)
			organoidData = []
			# Get all of the data for each variable name that was given
			for colVarName in varNames:
				# Get the Series index Number and the organoid label from the filename
				number = csvFilename.split('/', -1)
				number = number[-1]
				number = re.sub('[^0-9]', '', number)
				number = int(number)
				organoidLabel = csvFilename.split('/', -1)
				organoidLabel = organoidLabel[-1]			
				organoidLabel = re.sub('[^A-H]', '', organoidLabel)
				# Get and save the data
				csvData = MRLcv.getCSVcolData(colVarName, csvFilename)
				organoidData.append(csvData)

			organoidData = np.array(organoidData)
			organoidData = np.transpose(organoidData)
			organoidData = organoidData.tolist()
			organoidData = [[number, organoidLabel] +
							row for row in organoidData]
			# allOrganoidData.append(organoidData)
			allOrganoidData += organoidData
			# writer.writerows(organoidData)

		# f.close()
		return allOrganoidData

	def getAllSeriesDatasetPixelScale():
		"""No parameters for this function because it's only for project 2"""
		varNameList = ['Frame', 'X', 'Y', 'Slice', 'Length', 'FeretAngle']
		inputCSVdir = '../Original Data | Fiji Organoid Measurements/CSV Data/'

		allOrganoidData = MRLcv.getAllSeriesDataset(varNameList, inputCSVdir)
		um2pixel_factor = 512 / 653.17 # micrometers converted to pixels
		allOrganoidData = MRLcv.scaleDatasetVar(allOrganoidData, um2pixel_factor, 'X')
		allOrganoidData = MRLcv.scaleDatasetVar(allOrganoidData, um2pixel_factor, 'Y')
		if len(allOrganoidData) == 0:
			print('warning allOrganoidData is 0')
		return allOrganoidData

	@staticmethod
	def genAllSeriesCSVfile(outputCSVfilename: str, varNames: [str], inputCSVdir: str = "./", outputCSVDir: str = './'):
		""" THIS WILL ONLY WORK IF THE ORGANOID LABELS ARE BETWEEN A AND H. LOOK AT
		HOW I IMPLEMENTED THE WAY LABELS FOR NUMBERS AND ORGANOID_LABELS ARE SETUP"""
		# Reading the CSV measurement filenames into a list
		csvFilenameList = glob.glob(inputCSVdir + "*.csv")

		# The new CSV file to be created
		f = open(outputCSVDir + outputCSVfilename, 'w')
		writer = csv.writer(f)

		# Make the first row labels for each column
		fileColLabels = ['seriesIndex'] + ['organoidLabel'] + varNames
		writer.writerow(fileColLabels)
		allOrganoidData = []
		allOrganoidData.append(fileColLabels)

		# Process each CSV file with organoid data in it
		for csvFilename in csvFilenameList:
			# print(csvFilename)
			organoidData = []
			# Get all of the data for each variable name that was given
			for colVarName in varNames:
				# Get the Series index Number and the organoid label from the filename
				number = csvFilename.split('/', -1)
				number = number[-1]
				number = re.sub('[^0-9]', '', number)
				number = int(number)
				organoidLabel = csvFilename.split('/', -1)
				organoidLabel = organoidLabel[-1]			
				organoidLabel = re.sub('[^A-H]', '', organoidLabel)
				# Get and save the data
				csvData = MRLcv.getCSVcolData(colVarName, csvFilename)
				organoidData.append(csvData)

			organoidData = np.array(organoidData)
			organoidData = np.transpose(organoidData)
			organoidData = organoidData.tolist()
			organoidData = [[number, organoidLabel] +
							row for row in organoidData]
			# allOrganoidData.append(organoidData)
			allOrganoidData += organoidData
			writer.writerows(organoidData)

		f.close()
		return allOrganoidData

	@staticmethod
	def saveAllSeriesCSVfile(allSeriesCSVFilename: str, organoidDataSet):
		f = open(allSeriesCSVFilename, 'w')
		writer = csv.writer(f)
		# organoidDataSet = str(organoidDataSet)
		writer.writerows(organoidDataSet)
		f.close()

	@staticmethod
	def	getOrganoidData(allSeriesOrganoidData, seriesNum:int, organoidLabel:str, varName:str):
		"""Extracts the organoid data for a particular variable and organoid and returns a list of that variables values"""
		firstRowSkipped = False
		organoidData = []
		for row in allSeriesOrganoidData:
			if not firstRowSkipped:
				seriesNumIndex = row.index('seriesIndex')
				organoidLabelIndex = row.index('organoidLabel')
				varIndex = row.index(varName)
				firstRowSkipped = True
				continue
			if row[seriesNumIndex] == int(seriesNum) and row[organoidLabelIndex] == organoidLabel:
				organoidData.append(row[varIndex])
		return organoidData


	@staticmethod
	def scaleDatasetVar(organoidDataSet, scaleConversionFactor, label):
		"""Changes x y data from micro-meters to pixels
				assumes a format of [series#, organoidLabel, t, x, y, z, ...]
		"""

		firstRowSkipped = False
		for organoidData in organoidDataSet:
			if not firstRowSkipped:
				firstRowSkipped = True
				# print(organoidData.index(label))
				index = organoidData.index(label)
				continue		
			organoidData[index] = str(
				float(organoidData[index]) * scaleConversionFactor)
		return organoidDataSet

	@staticmethod
	def getLabelIndex(dataSet, label:str):
		for row in dataSet:
			return row.index(label)

	@staticmethod
	def getZtrackedOrganoidSeriesArr(seriesArr, orgZ):
		"""ASSUMES the input is a BW video in [t, z, x, y]"""
		timeRange, zRange, xRange, yRange = seriesArr.shape
		zTrackedSeriesArr = None
		for time in range(timeRange):
			zPlaneIndex = int(orgZ[time]) - 1 
			frame = seriesArr[time, zPlaneIndex, : , :]
			zTrackedSeriesArr = MRLcv.addFrame2video(zTrackedSeriesArr, frame)
		return zTrackedSeriesArr

	@staticmethod 
	def getOrganoidLabels(allOrganoidData, seriesNum):

		seriesIndex = MRLcv.getLabelIndex(allOrganoidData, 'seriesIndex')
		organoidLabelIndex = MRLcv.getLabelIndex(allOrganoidData, 'organoidLabel')
		organoidLabels = set([])
		for row in allOrganoidData:
			if row[seriesIndex] == int(seriesNum):
				organoidLabels.add(row[organoidLabelIndex])
		organoidLabels = list(organoidLabels)
		organoidLabels.sort()
		return organoidLabels
	
	# GENERAL HELPER FUNCTIONS 		_________________________________________________
	@staticmethod
	def linSpaceList(beg, end, num: int, dataType=1.0):
		"""By default returns floating point data in lists"""
		return [type(dataType)(i) for i in np.linspace(beg, end, num)]

	# NEW METHODS___________________________________________________________________
	
	# def removeLabelsFromDataset(allOrganoidData):
	# 	for row in allOrganoidData:

	@staticmethod
	def getDatasetColData(allOrganoidDataset, varName:str, series:int=None, label:str=None):
		dataset = []
		varIndex = MRLcv.getLabelIndex(allOrganoidDataset, varName)
		seriesIndex = MRLcv.getLabelIndex(allOrganoidDataset, 'seriesIndex')
		labelIndex = MRLcv.getLabelIndex(allOrganoidDataset, 'organoidLabel')
		# labels = MRLcv.getOrganoidLabels(allOrganoidDataset, series)

		#remove the labels
		# allOrganoidDataset.pop(0)


		labelingRow = next(iter(allOrganoidDataset))
		#Get all of the data for that index
		for row in allOrganoidDataset:
			if row == labelingRow:
				continue
			#All data
			if series is None and label is None:
				dataset.append(float(row[varIndex]))
			#All data for one series
			elif series == row[seriesIndex] and label is None:
				dataset.append(float(row[varIndex]))
			#All data for one organoid
			elif series == row[seriesIndex] and label == row[labelIndex]:
				dataset.append(float(row[varIndex]))

		return dataset
		
	@staticmethod
	def makeDir(directory:str):
		"""Make the directory provided if the directory does not already exist
		"""
		if not os.path.exists(directory):
			os.makedirs(directory)
		return os.path.exists(directory)
	
	@staticmethod
	def getDistance(x1, y1, x2, y2):
		"""assumes all list parameters are the same length """
		dist = [math.sqrt((x2[t] - x1[t])**2 + (y2[t] - y1[t])**2) for t in range(0, len(x1))]  
		return dist

	@staticmethod
	def gui_drawRectangle():
		# Lists to store the bounding box coordinates
		top_left_corner=[]
		bottom_right_corner=[]

		# function which will be called on mouse input
		def drawRectangle(action, x, y, flags, *userdata):
			# Referencing global variables 
			global top_left_corner, bottom_right_corner
			# Mark the top left corner when left mouse button is pressed
			if action == cv2.EVENT_LBUTTONDOWN:
				top_left_corner = [(x,y)]
				# When left mouse button is released, mark bottom right corner
			elif action == cv2.EVENT_LBUTTONUP:
				bottom_right_corner = [(x,y)]    
				# Draw the rectangle
				cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
				cv2.imshow("Window",image)

		# Read Images
		image = cv2.imread("../foo/spongebob.jpg")
		# Make a temporary image, will be useful to clear the drawing
		temp = image.copy()
		# Create a named window
		cv2.namedWindow("Window")
		# highgui function called when mouse events occur
		cv2.setMouseCallback("Window", drawRectangle)

		k=0
		# Close the window when key q is pressed
		while k!=113:
			# Display the image
			cv2.imshow("Window", image)
			k = cv2.waitKey(0)
			# If c is pressed, clear the window, using the dummy image
			if (k == 99):
				image= temp.copy()
				cv2.imshow("Window", image)

		cv2.destroyAllWindows()

	@staticmethod
	def	gui_annotateWithMouse():

		maxScaleUp = 100
		scaleFactor = 1
		windowName = "Resize Image"
		trackbarValue = "Scale"

		# read the image
		image = cv2.imread("../foo/spongebob.jpg")

		# Create a window to display results and  set the flag to Autosize
		cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

		# Callback functions
		def scaleImage(*args):
			# Get the scale factor from the trackbar 
			scaleFactor = 1+ args[0]/100.0
			# Resize the image
			scaledImage = cv2.resize(image, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
			cv2.imshow(windowName, scaledImage)

		# Create trackbar and associate a callback function
		cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImage)

		# Display the image		
		cv2.imshow(windowName, image)
		c = cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	@staticmethod
	def gui_trackbar():
		alpha_slider_max = 100
		title_window = 'Linear Blend'
		def on_trackbar(val):
			alpha = val / alpha_slider_max
			beta = ( 1.0 - alpha )
			dst =cv2.addWeighted(src1, alpha, src2, beta, 0.0)
			cv.imshow(title_window, dst)
		parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
		parser.add_argument('--input1', help='Path to the first input image.', default='LinuxLogo.jpg')
		parser.add_argument('--input2', help='Path to the second input image.', default='WindowsLogo.jpg')
		args = parser.parse_args()
		src1 = cv2.imread(cv2.samples.findFile(args.input1))
		src2 = cv2.imread(cv2.samples.findFile(args.input2))
		if src1 is None:
			print('Could not open or find the image: ', args.input1)
			exit(0)
		if src2 is None:
			print('Could not open or find the image: ', args.input2)
			exit(0)
		cv2.namedWindow(title_window)
		trackbar_name = 'Alpha x %d' % alpha_slider_max
		cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)
		# Show some stuff
		on_trackbar(0)
		# Wait until user press some key
		cv2.waitKey()

#Program: Start_________________________________________________________________

def videoProcessingPipeline():
	print("Program is initiated...\n")
	# snap = memoryTrace(label='StartProgram')
	# Parameters | Batch Video Processing parameters
	firstSeriesNumber = 4  # TIF SERIES FILENAMES START AT 1
	lastSeriesNumber = 25
	numVideosPerSeries = 4**0
	makeNonGridVideos = True
	makeCleanVideos = True
	makeTimeConcatVideos = True
	makeKeypointVideos = True 
	makeOrganoidVideos = True
	FPS = 2

	# Names | Name of the experiment (video experiment to optimize parameters)__
	experiementName = "usingStats2optimizeParams_02_testIfCodeCompiles"
	# Names | Names of the directories to read/write files
	tifFileReadDir = '../Original Data | Nikon Imagine Tiff Files/'
	saveDir = "../Processed Data | Video Experiments "
	# Directories ______________________________________________________________
	allSeriesSaveDir = saveDir + experiementName + "/allSeries/"
	seriesSaveDir = saveDir + experiementName + "/series/"
	seriesKPSaveDir = seriesSaveDir + 'grid_keypoints/'
	seriesNoGridSaveDir = seriesSaveDir + "noGrid_orig/"
	seriesNoGridKPSaveDir = seriesSaveDir + "noGrid_keypoints/"
	# Names | Processed Video files (output)
	seriesSaveFilename = "_" + experiementName
	allSeriesSaveFilename = "allSeries_" + experiementName
	seriesKPSaveFilename = "_" + experiementName + '_keypoints'
	seriesNoGridSaveFilename = "_" + experiementName + '_noGrid'
	seriesNoGridKPSaveFilename = seriesNoGridSaveFilename + "_keypoints"
	allSeriesKPSaveFilename = "allSeries_" + experiementName + '_keypoints'
	#Get CSV Data from measurements
	varNameList = ['Frame', 'X', 'Y', 'Slice', 'Length', 'FeretAngle']
	inputCSVdir = '../Original Data | Fiji Organoid Measurements/CSV Data/'
	
	# Time Concatenation Videos of all Series___________________________________
	allSeriesInTime = None
	allSeriesInTime_kp = None
	# Get Organoid Dataset
	allOrganoidData = MRLcv.getAllSeriesDataset(varNameList, inputCSVdir)
	um2pixel_factor = 512 / 653.17 # micrometers converted to pixels
	allOrganoidData = MRLcv.scaleDatasetVar(allOrganoidData, um2pixel_factor, 'X')
	allOrganoidData = MRLcv.scaleDatasetVar(allOrganoidData, um2pixel_factor, 'Y')
	for series in range(firstSeriesNumber, lastSeriesNumber + 1):
		# Filenames | names of files for each respective video series
		seriesNum = str(series)
		seriesName = 'series' + seriesNum.zfill(2)
		tiffFilename = 'series' + seriesNum.zfill(2) + '.tif'
		# Open Files | the series file to be processed
		print("\nProcessing: " + tiffFilename)
		series3Darray = MRLcv.readTifFile(tifFileReadDir + tiffFilename)
		# Pre-Processing | convert to 8bit, so that openCV functions work
		series3Darray = MRLcv.uint_to_uint8(series3Darray)


		# Iterate through Each Organoid in each Series TIF file
		organoidLabels = MRLcv.getOrganoidLabels(allOrganoidData, seriesNum)
		for organoidLabel in organoidLabels:
			organoidName = 'organoid' + organoidLabel
			print('organoidLabel = ' + organoidLabel)

			# Data | missing data
			if int(seriesNum) == 1 and organoidLabel =='A': continue 
			# Video | Format | z-tracked series videos for each organoid
			orgY = MRLcv.getOrganoidData(allOrganoidData, seriesNum, organoidLabel, 'Y')
			orgX = MRLcv.getOrganoidData(allOrganoidData, seriesNum, organoidLabel, 'X')
			orgZ = MRLcv.getOrganoidData(allOrganoidData, seriesNum, organoidLabel, varName='Slice')
			zTrackedOrganoidArr = MRLcv.getZtrackedOrganoidSeriesArr(series3Darray, orgZ)		
			# Video | Format | add color channels to the B&W video, so color videos can be concatenated
			zTrackedOrganoidArr = MRLcv.addColorChannels2bwVideo(zTrackedOrganoidArr)

			# Setup | Make a list of copies of the series video, so each can be processed uniquely
			videos = [zTrackedOrganoidArr.copy() for _ in range(numVideosPerSeries)]
			stdDev = 16
			mean = 83
			# Setup | Parameters | Static
			thresholdParams = MRLcv.getIterativeParams('min', numVideosPerSeries, int(mean - 2 * stdDev), int(mean + 2 * stdDev), 5)
			annoteParams = MRLcv.getIterativeParams('')
			fftParams = MRLcv.getIterativeParams('freqMin', numVideosPerSeries, 0.0, 0.33)
			cannyParams = MRLcv.getIterativeParams('thres1', numVideosPerSeries, 0, 120)
			# Setup | Parameters | Variable
			thresholdParams = MRLcv.setNewDefaultParam('thresFilter', cv2.THRESH_BINARY)
			gausParams = MRLcv.setNewDefaultParam('kernalSize', (3,3), numVideosPerSeries)
			gausParams = MRLcv.setNewDefaultParam('sigmaX', 1.0, numVideosPerSeries, gausParams)
			gausParams = MRLcv.setNewDefaultParam('sigmaY', 1.0, numVideosPerSeries, gausParams)
			houghesParams = MRLcv.getIterativeParams()


			# Pre-Processing | Prepare files so cannyEdgeDetection is more effective________________
			# videos = MRLcv.videoProcessor(videos, "gaussian", gausParams)
			videos = MRLcv.videoProcessor(videos, "cannyEdgeDetector", cannyParams)
			# videos = MRLcv.videoProcessor(videos, "addFFT2video", fftParams)
			# videos = MRLcv.videoProcessor(videos, "addThreshold2Video", thresholdParams)
			# videos = MRLcv.videoProcessor(videos, 'addHoughesCircles', houghesParams )
			# processParams, processVarName = fftParams, 'freqMin'	
			# processParams, processVarName = thresholdParams, 'min'
			processParams, processVarName = cannyParams, 'thres1'

			# Setup | Keypoint Parameters___________________________________________________________
			keypointParams = MRLcv.getKeypointParams(numVideosPerSeries)

			# Data | Keypoints______________________________________________________________________
			# keypointDataset = MRLcv.getVideoKeypoints(video4, blobDetParams) # THE NON BATCH PROCESS

			# annotation | keypoints________________________________________________________________
			videos_kp = MRLcv.videoProcessor(videos, "addKeypoints2video", keypointParams) if makeKeypointVideos else None
			
			# annotation | text ____________________________________________________________________
			annoteParams = MRLcv.getAnnoteParam_withParamValue(annoteParams, processParams, processVarName)
			# videos = MRLcv.annotateParamValue2videos(videos, annoteParams, processParams, processVarName)	
			videos_kp = MRLcv.annotateParamValue2videos(videos_kp, annoteParams, processParams, processVarName) if makeKeypointVideos else None

			# annotation | Organoid Tracker_________________________________________________
			loc = (orgX, orgY)
			# videos = [ MRLcv.addVideoOrganoidTrackerDot(video, organoidLabel, loc) for video in videos]
			videos_kp = [ MRLcv.addVideoOrganoidTrackerDot(video, organoidLabel, loc) for video in videos_kp] if makeKeypointVideos else None

			# Video | Saving Processed AVI Files____________________________________________________
			if makeNonGridVideos: 
				MRLcv.saveAllVideos(videos, seriesName + '_' + organoidLabel + seriesNoGridSaveFilename, seriesNoGridSaveDir, FPS)
			if makeCleanVideos:
				videos = MRLcv.videoProcessor(videos, "concat2videoGrid", {})
				MRLcv.saveAllVideos(videos, seriesName+ '_' + organoidLabel  + seriesSaveFilename, seriesSaveDir + 'grid_orig/', FPS)
			if makeTimeConcatVideos:
				allSeriesInTime = MRLcv.concatAllVideosTime(allSeriesInTime, videos)
			if makeKeypointVideos:
				MRLcv.saveAllVideos(videos_kp, seriesName+ '_' + organoidLabel  + seriesNoGridKPSaveFilename, seriesNoGridKPSaveDir, FPS)
				videos_kp = MRLcv.videoProcessor(videos_kp, "concat2videoGrid", {})
				MRLcv.saveAllVideos(videos_kp, seriesName+ '_' + organoidLabel  + seriesKPSaveFilename, seriesKPSaveDir, FPS)
				allSeriesInTime_kp = MRLcv.concatAllVideosTime(allSeriesInTime_kp, videos_kp)

		# Saving Processed Files | Save all of the serie∆∆s files concatenated in time
		if makeOrganoidVideos:
			MRLcv.saveAllVideos(allSeriesInTime, seriesName+ '_' + 'allOrganoids', seriesSaveDir)
			if makeKeypointVideos:
				MRLcv.saveAllVideos(allSeriesInTime_kp, seriesName+ '_' + 'allOrganoidsKP', seriesSaveDir, FPS)

	# Saving Processed Files | Save all of the series files concatenated in time
	if makeTimeConcatVideos:
		MRLcv.saveAllVideos(allSeriesInTime, allSeriesSaveFilename, allSeriesSaveDir)
	if makeKeypointVideos:
		MRLcv.saveAllVideos(allSeriesInTime_kp, allSeriesKPSaveFilename, allSeriesSaveDir, FPS)
	print("\n...Program was successful!")



def memoryTrace(label:str='', objs=[], snap=None):
	tracemalloc.start() #start this code earlier, somewhere in parent call
	skip = True 
	if skip: return None
	### Run application

	# for obj in objs:
	# 	print('size_' + str(sys.getsizeof(obj)) + '_ID_' + str(id(obj)))


	spacer = "\t"
	### take memory usage snapshot
	snapshot = tracemalloc.take_snapshot()
	top_stats = snapshot.statistics('lineno')

	### Print top 10 files allocating the most memory
	print(spacer + "[Top 10]_LOCATION_" + label)
	for stat in top_stats[:10]:
		print(spacer + str(stat))
		#Show Memory trace for each item taking up memory
		print(spacer*2 + "%s memory blocks: %.1f KiB_________" % (stat.count, stat.size / 1024))
		for line in stat.traceback.format():
			print(spacer*2 + 'traceback_' + str(line))
	#Comparison to snapshot
	if snap is not None:
		top_stats = snapshot.compare_to(snap, 'lineno')
		#Differences
		print(spacer + "[ Top 10 differences]")
		for stat in top_stats[:10]:
			print(spacer + 'diff_' + str(stat))

	print('')
	return snapshot

def genAllSeriesOrganoidCSVfile():
	varNameList = ['Frame', 'X', 'Y', 'Slice', 'Length', 'FeretAngle']
	print(varNameList)
	inputCSVdir = '../originalData_fijiOrganoidMeasurements/measurements_CSVdata/'
	outputCSVdir = '../processedData_CSVandExcel/'
	outputCSVfilename = "allSeriesDataset.csv"
	allOrganoidData = MRLcv.genAllSeriesCSVfile(outputCSVfilename, varNameList, inputCSVdir, outputCSVdir)


	um2pixel_factor = 512 / 653.17 # micrometers converted to pixels
	allOrganoidData = MRLcv.scaleDatasetVar(allOrganoidData, um2pixel_factor, 'X')
	allOrganoidData = MRLcv.scaleDatasetVar(allOrganoidData, um2pixel_factor, 'Y')
	# allOrganoidData = MRLcv.changeScaleSettings(allOrganoidData, um2pixel_factor, 'Slice')
	
	organoid12A_X = MRLcv.getOrganoidData(allOrganoidData, 12, 'A', 'Slice')
	print(*organoid12A_X, sep='\n')
	MRLcv.saveAllSeriesCSVfile(
		outputCSVdir + "allSeriesDataset_XYZ.csv", allOrganoidData)

def getDatasetStats(allOrganoidDataset, varName:str , varMin:int, varMax:int, numBins:int=15, saveDir:str='./', filename:str=None):
	# dataset = []
	# index = MRLcv.getLabelIndex(allOrganoidDataset, varName)
	# print(index)
	
	# #remove the labels
	# allOrganoidDataset.pop(0)

	# #Get all of the data for that index
	# for row in allOrganoidDataset:
	# 	dataset.append(float(row[index]))
	# 	print(row[index])		#testing

	firstSeriesIndex = 1
	lastSeriesIndex = 25

	if varName == 'XYdistance':
		dataset = []
		for seriesIndex in range (firstSeriesIndex, lastSeriesIndex + 1):
			labels = MRLcv.getOrganoidLabels(allOrganoidDataset, seriesIndex)
			labelIter = iter(labels)
			for label in labelIter:
				#calculate distance between each and store data
				labelIter2 = copy.deepcopy(labelIter)
				for otherLabel in labelIter2:
					x1 = MRLcv.getDatasetColData(allOrganoidDataset, 'X', seriesIndex, label)
					x2 = MRLcv.getDatasetColData(allOrganoidDataset, 'X', seriesIndex, otherLabel)
					y1 = MRLcv.getDatasetColData(allOrganoidDataset, 'Y', seriesIndex, label)
					y2 = MRLcv.getDatasetColData(allOrganoidDataset, 'Y', seriesIndex, otherLabel)
					distance = MRLcv.getDistance(x1, y1, x2, y2)
					dataset += distance
	else:
		dataset = MRLcv.getDatasetColData(allOrganoidDataset, varName)
	

	stats = {}
	stats['mean'] = np.mean(dataset)
	stats['median'] = np.median(dataset)
	stats['stdDev'] = np.std(dataset)
	stats['min'] = np.min(dataset)
	stats['max'] = np.max(dataset)

	for stat in stats:
		print(stat + ' of ' + str(varName) + " is " + str(stats[stat]))
	print('N = ' + str(len(dataset)))
	print('👽 μ - 2𝜎 = ' + str(stats['mean'] - 2 * stats['stdDev']))
	print('👽 μ + 2𝜎 = ' + str(stats['mean'] + 2 * stats['stdDev']))

	# Creating histogram
	fig, ax = plt.subplots(figsize =(10, 7))
	bins = np.linspace(varMin, varMax, numBins) #Range of the stat variable
	ax.hist(dataset, bins)
	# Show plot
	plt.title('Histograph of Organoid Parameter: ' + varName)
	ax.set_xticks(bins)
	# plt.show()
	
	MRLcv.makeDir(saveDir)
	if filename is None:
		plotName = saveDir + 'histogram_' + str(varName) + '_bins' + str(numBins) + '.jpg'
		plt.savefig(plotName)
		print("stat plot saved: " + plotName)
	else:
		plt.savefig(saveDir + filename + '.jpg')

def binaryThres_blackIsTheNewGrey():
	for series in range(firstSeriesNumber, lastSeriesNumber + 1):
		# Filenames | names of files for each respective video series
		seriesNum = str(series)
		seriesName = 'series' + seriesNum.zfill(2)
		tiffFilename = 'series' + seriesNum.zfill(2) + '.tif'
		# Open Files | the series file to be processed
		print("\nProcessing: " + tiffFilename)
		series3Darray = MRLcv.readTifFile(tifFileReadDir + tiffFilename)
		# Pre-Processing | convert to 8bit, so that openCV functions work
		series3Darray = MRLcv.uint_to_uint8(series3Darray)

def hougesCirclesExperiements():			
	img = cv2.imread('circles.jpg', cv2.IMREAD_COLOR)
	MRLcv.showImage(img, '', 0)
	MRLcv.addHoughesCircles2img_COPYPASTE()
	MRLcv.addHoughesCircles2img(img)
	return "just a test function"


# videoProcessingPipeline()

statDir = '../Processed Data | Statistics/'
getDatasetStats(MRLcv.getAllSeriesDatasetPixelScale(), 'Length', 0, 512, 15, statDir)
getDatasetStats(MRLcv.getAllSeriesDatasetPixelScale(), 'XYdistance', 0, 493, 100, statDir)

MRLcv.gui_drawRectangle()
MRLcv.gui_annotateWithMouse()
MRLcv.gui_trackbar()

