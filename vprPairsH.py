import numpy as np
import scipy
from tqdm import tqdm
import csv
import os
import re
from sys import exit, argv
import time
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

import pydegensac


def readQuery(file):
	queryPairs = []

	with open(file) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')

		for i, row in enumerate(csvReader):
			if(i==0):
				continue
			else:
				queryPairs.append(row)

	print("Query read.")

	return queryPairs


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def loadFeat(queryPairs, frontDir, rearDir):
		frontDict = {}
		rearDict = {}

		for row in tqdm(queryPairs, total=len(queryPairs)):
			frontImg = row[0]

			frontFeat = frontImg.replace('png', 'd2-net')
			frontDict[os.path.basename(frontImg).replace('.png', '')] = np.load(frontFeat)

		rearFeats = natural_sort([file for file in os.listdir(rearDir) if '.d2-net' in file])
		rearFeatsFull = [os.path.join(rearDir, feat) for feat in rearFeats]
		
		for i in tqdm(range(len(rearFeats)), total=len(rearFeats)):
			rearDict[rearFeats[i].replace('.d2-net', '')] = np.load(rearFeatsFull[i])

		print("Features loaded.")

		return frontDict, rearDict


def numInliers2(feat1, feat2):
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(feat1['descriptors'], feat2['descriptors'])
	matches = sorted(matches, key=lambda x:x.distance)

	match1 = [m.queryIdx for m in matches]
	match2 = [m.trainIdx for m in matches]

	keypoints_left = feat1['keypoints'][match1, : 2]
	keypoints_right = feat2['keypoints'][match2, : 2]

	np.random.seed(0)

	# model, inliers = ransac(
	# 	(keypoints_left, keypoints_right),
	# 	AffineTransform, min_samples=4,
	# 	residual_threshold=8, max_trials=10000
	# )
	
	H, inliers = pydegensac.findHomography(keypoints_left, keypoints_right, 8.0, 0.99, 10000)

	n_inliers = np.sum(inliers)

	return n_inliers


def getPairs(queryPairs, frontDict, rearDict, rearDir):
	matches = []

	rearImgs = natural_sort([file for file in os.listdir(rearDir) if '.png' in file])
	rearImgs = [file.replace('.png', '') for file in rearImgs]

	for pair in tqdm(queryPairs, total=len(queryPairs)):
		frontImg = pair[0]
		frontFeat = frontDict[os.path.basename(frontImg).replace('.png', '')]

		dbStart = os.path.basename(pair[3]).replace('.png', '')
		dbEnd = os.path.basename(pair[4]).replace('.png', '')

		dbStartIdx = rearImgs.index(dbStart)
		dbEndIdx = rearImgs.index(dbEnd)

		maxInliers = -100
		maxRear = None

		# for idx in tqdm(range(dbStartIdx, dbEndIdx+1)):
		for idx in range(dbStartIdx, dbEndIdx+1):
			rearFeat = rearDict[rearImgs[idx]]

			inliers = numInliers2(frontFeat, rearFeat)
			
			if(maxInliers < inliers):
				maxInliers = inliers
				maxRear = rearImgs[idx]

		matches.append([frontImg, maxRear, str(maxInliers)])

	return matches


def writeMatches(matches):
	with open('dataGenerate/vprOutputH.csv', 'w', newline='') as file:
		writer = csv.writer(file)

		title = ['FrontImage', 'RearImage', 'Correspondences']
		writer.writerow(title)

		for match in matches:
			writer.writerow(match)


if __name__ == '__main__':
	pairsFile = argv[1]
	frontDir = argv[2]
	rearDir = argv[3]

	queryPairs = readQuery(pairsFile)

	frontDict, rearDict = loadFeat(queryPairs, frontDir, rearDir)

	matches = getPairs(queryPairs, frontDict, rearDict, rearDir)

	writeMatches(matches)