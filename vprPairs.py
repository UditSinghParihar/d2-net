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

		print("Features loaded.")

		return frontDict, rearDict


def numInliers(feat1, feat2):
	matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)

	keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
	keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
	np.random.seed(0)

	model, inliers = ransac(
		(keypoints_left, keypoints_right),
		AffineTransform, min_samples=4,
		residual_threshold=12, max_trials=500
	)

	n_inliers = np.sum(inliers)

	return n_inliers, matches


def getPairs(probPairs, frontDict, rearDict):
	matches = []

	for pair in tqdm(probPairs, total=len(probPairs)):
		frontImg = pair[0]
		frontFeat = frontDict[frontImg]

		maxInliers = -100
		maxIdx = -1

		for i in range(1, len(pair)):
			rearImg = pair[i]
			rearFeat = rearDict[rearImg]
	
			inliers, denseMatches = numInliers(frontFeat, rearFeat)
			print("Inliers:", inliers, denseMatches.shape)
			
			if(maxInliers < inliers):
				maxInliers = inliers
				maxIdx = i

		match = (frontImg, pair[maxIdx], maxInliers)
		print(match)
		matches.append(match)

	return matches


if __name__ == '__main__':
	pairsFile = argv[1]
	frontDir = argv[2]
	rearDir = argv[3]

	probPairs = readPairs(pairsFile)

	frontDict, rearDict = loadFeat(probPairs, frontDir, rearDir)

	getPairs(probPairs, frontDict, rearDict)


