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


def writeMatches(matches):
	with open('dataGenerate/vprOutputHSubSIFT.csv', 'w', newline='') as file:
		writer = csv.writer(file)

		title = ['FrontImage', 'RearImage', 'Correspondences']
		writer.writerow(title)

		for match in matches:
			writer.writerow(match)


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def numInliers(img1, img2):
	# surf = cv2.xfeatures2d.SURF_create(100)
	surf = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = surf.detectAndCompute(img1, None)
	kp2, des2 = surf.detectAndCompute(img2, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	for m, n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)
	
	if(src_pts.shape[0] < 5):
		return src_pts.shape[0]

	# model, inliers = ransac(
	# 		(src_pts, dst_pts),
	# 		AffineTransform, min_samples=4,
	# 		residual_threshold=8, max_trials=10000
	# 	)

	H, inliers = pydegensac.findHomography(src_pts, dst_pts, 8.0, 0.99, 10000)

	n_inliers = np.sum(inliers)

	# inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
	# inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
	# placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
	# image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)

	# cv2.imshow('Matches', image3)
	# cv2.waitKey(1000)

	return n_inliers


def getPairs(queryPairs, rearDir):
	matches = []

	rearImgs = natural_sort([file for file in os.listdir(rearDir) if '.png' in file])
	rearImgs = [file.replace('.png', '') for file in rearImgs]

	progressBar = tqdm(queryPairs, total=len(queryPairs))
	correct = 0.0
	total = 0.0
	for pair in progressBar:
		frontFile = pair[0]
		
		gtStId = int(os.path.basename(pair[1]).replace('.png', ''))
		gtEndId = int(os.path.basename(pair[2]).replace('.png', ''))

		frontImg = np.array(Image.open(frontFile).convert('L').resize((400, 400)))
		frontImg = frontImg[:, :, np.newaxis]
		frontImg = np.repeat(frontImg, 3, -1)

		dbStart = os.path.basename(pair[3]).replace('.png', '')
		dbEnd = os.path.basename(pair[4]).replace('.png', '')

		dbStartIdx = rearImgs.index(dbStart)
		dbEndIdx = rearImgs.index(dbEnd)

		maxInliers = -100
		maxRear = None

		# for idx in tqdm(range(dbStartIdx, dbEndIdx+1)):
		for idx in range(dbStartIdx, dbEndIdx+1):
			rearFile = os.path.join(rearDir, rearImgs[idx]+".png")

			rearImg = np.array(Image.open(rearFile).convert('L').resize((400, 400)))
			rearImg = rearImg[:, :, np.newaxis]
			rearImg = np.repeat(rearImg, 3, -1)

			inliers = numInliers(frontImg, rearImg)

			if(maxInliers < inliers):
				maxInliers = inliers
				maxRear = rearImgs[idx]

		matches.append([frontFile, maxRear, str(maxInliers)])
		total += 1.0
		
		if(gtStId < int(maxRear) < gtEndId):
			correct += 1.0
		progressBar.set_postfix(retrieve=('{}/{}({:.2f})'.format(correct, total, correct*100.0/total)))

	return matches


if __name__ == '__main__':
	pairsFile = argv[1]
	frontDir = argv[2]
	rearDir = argv[3]

	queryPairs = readQuery(pairsFile)

	matches = getPairs(queryPairs, rearDir)

	writeMatches(matches)
