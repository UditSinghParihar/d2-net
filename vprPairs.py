# import argparse
import numpy as np
import scipy
from tqdm import tqdm
import csv
import os
from sys import exit, argv
import torch

import imageio
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform


def readPairs(file):
	probPairs = []

	with open(file) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')

		for i, row in enumerate(csvReader):
			if(i==0):
				continue
			else:
				probPairs.append(row)

	print("Pairs read.")

	return probPairs


# def extract(image, model, device, preprocessing, multiscale):
# 	if len(image.shape) == 2:
# 		image = image[:, :, np.newaxis]
# 		image = np.repeat(image, 3, -1)

# 	resized_image = image
	
# 	fact_i = image.shape[0] / resized_image.shape[0]
# 	fact_j = image.shape[1] / resized_image.shape[1]

# 	input_image = preprocess_image(
# 		resized_image,
# 		preprocessing=preprocessing
# 	)
# 	with torch.no_grad():
# 		if multiscale:
# 			keypoints, scores, descriptors = process_multiscale(
# 				torch.tensor(
# 					input_image[np.newaxis, :, :, :].astype(np.float32),
# 					device=device
# 				),
# 				model
# 			)
# 		else:
# 			keypoints, scores, descriptors = process_multiscale(
# 				torch.tensor(
# 					input_image[np.newaxis, :, :, :].astype(np.float32),
# 					device=device
# 				),
# 				model,
# 				scales=[1]
# 			)

# 	keypoints[:, 0] *= fact_i
# 	keypoints[:, 1] *= fact_j
# 	keypoints = keypoints[:, [1, 0, 2]]

# 	feat = {}
# 	feat['keypoints'] = keypoints
# 	feat['scores'] = scores
# 	feat['descriptors'] = descriptors

# 	return feat


# def extractFeat(probPairs):
# 	use_cuda = torch.cuda.is_available()
# 	device = torch.device("cuda:0" if use_cuda else "cpu")
# 	weights = '/home/udit/d2-net/checkpoints/checkpoint_road_more/d2.15.pth'
# 	preprocessing = 'caffe'
# 	multiscale = False

# 	model = D2Net(
# 		model_file=weights,
# 		use_relu=True,
# 		use_cuda=use_cuda
# 	)

# 	frontDict = {}
# 	rearDict = {}

	# for pair in tqdm(probPairs, total=len(probPairs)):
	# 	front = pair[0]

	# 	if(front not in frontDict):
	# 		image = np.array(Image.open(front).convert('L'))
	# 		image = image[:, :, np.newaxis]
	# 		image = np.repeat(image, 3, -1)

	# 		frontDict[front] = extract(image, model, device, preprocessing, multiscale)

	# for pair in tqdm(probPairs, total=len(probPairs)):
	# 	for i in range(1, len(pair)):
	# 		rear = pair[i]

	# 		if(rear not in rearDict):
	# 			image = np.array(Image.open(rear).convert('L'))
	# 			image = image[:, :, np.newaxis]
	# 			image = np.repeat(image, 3, -1)

	# 			rearDict[rear] = extract(image, model, device, preprocessing, multiscale)

# 	print("Features extracted.")

# 	return frontDict, rearDict


def loadFeat(probPairs, frontDir, rearDir):
		frontDict = {}
		rearDict = {}

		for pair in tqdm(probPairs, total=len(probPairs)):
			frontImg = pair[0]

			if(frontImg not in frontDict):
				frontFeat = frontImg.replace('png', 'd2-net')
				frontDict[frontImg] = np.load(frontFeat)

		for pair in tqdm(probPairs, total=len(probPairs)):
			for i in range(1, len(pair)):
				rearImg = pair[i]

				if(rearImg not in rearDict):
					rearFeat = rearImg.replace('png', 'd2-net')
					rearDict[rearImg] = np.load(rearFeat)

		return frontDict, rearDict


if __name__ == '__main__':
	pairsFile = argv[1]
	frontDir = argv[2]
	rearDir = argv[3]

	probPairs = readPairs(pairsFile)

	frontDict, rearDict = loadFeat(probPairs, frontDir, rearDir)