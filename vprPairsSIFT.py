import numpy as np
import scipy
from tqdm import tqdm
import csv
import os
from sys import exit, argv
import time
import torch

import imageio
import cv2
from PIL import Image
import matplotlib.pyplot as plt


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


def draw(kp1, kp2, good, frontImg, rearImg):
	MIN_MATCH_COUNT = 1

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		draw_params = dict(matchColor = (0,255,0),
						   singlePointColor = None,
						   matchesMask = matchesMask,
						   flags = 2)
		img3 = cv2.drawMatches(frontImg,kp1,rearImg,kp2,good,None,**draw_params)
		cv2.imshow('Matches', img3)
		cv2.waitKey(0)

	else:
		print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		draw_params = dict(matchColor = (0,255,0),
						   singlePointColor = None,
						   matchesMask = matchesMask,
						   flags = 2)
		img3 = cv2.drawMatches(frontImg,kp1,rearImg,kp2,good,None,**draw_params)
		cv2.imshow('Matches', img3)
		cv2.waitKey(0)


def numInliers(frontImg, rearImg):
	surf = cv2.xfeatures2d.SURF_create()
	# surf = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = surf.detectAndCompute(frontImg, None)
	kp2, des2 = surf.detectAndCompute(rearImg, None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	for m, n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	# draw(kp1, kp2, good, frontImg, rearImg)

	n_inliers = len(good)

	return n_inliers, matches


def getPairs(probPairs):
	matches = []

	for pair in tqdm(probPairs, total=len(probPairs)):
		frontFile = pair[0]

		frontImg = np.array(Image.open(frontFile).convert('L').resize((400, 400)))
		frontImg = frontImg[:, :, np.newaxis]
		frontImg = np.repeat(frontImg, 3, -1)		

		maxInliers = -100
		maxIdx = -1

		for i in range(1, len(pair)):
			rearFile = pair[i]

			rearImg = np.array(Image.open(rearFile).convert('L').resize((400, 400)))
			rearImg = rearImg[:, :, np.newaxis]
			rearImg = np.repeat(rearImg, 3, -1)

			inliers, denseMatches = numInliers(frontImg, rearImg)
			# print("Inliers:", inliers, len(denseMatches))
		
			if(maxInliers < inliers):
				maxInliers = inliers
				maxIdx = i

		match = [frontFile, pair[maxIdx], str(maxInliers)]
		print(match)
		matches.append(match)

	return matches


def writeMatches(matches):
	with open('dataGenerate/vprOutputSIFT.csv', 'w', newline='') as file:
		writer = csv.writer(file)

		title = ['FrontImage', 'RearImage', 'Correspondences']
		writer.writerow(title)

		for match in matches:
			writer.writerow(match)


if __name__ == '__main__':
	pairsFile = argv[1]

	probPairs = readPairs(pairsFile)
	matches = getPairs(probPairs)

	print(matches)

	writeMatches(matches)
