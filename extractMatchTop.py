import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm
import time
import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

import cv2
import matplotlib.pyplot as plt
import os
from sys import exit, argv
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform


def extract(image, model, device, multiscale=False, preprocessing='caffe'):
	resized_image = image

	fact_i = image.shape[0] / resized_image.shape[0]
	fact_j = image.shape[1] / resized_image.shape[1]

	input_image = preprocess_image(
		resized_image,
		preprocessing=preprocessing
	)
	with torch.no_grad():
		if multiscale:
			keypoints, scores, descriptors = process_multiscale(
				torch.tensor(
					input_image[np.newaxis, :, :, :].astype(np.float32),
					device=device
				),
				model
			)
		else:
			keypoints, scores, descriptors = process_multiscale(
				torch.tensor(
					input_image[np.newaxis, :, :, :].astype(np.float32),
					device=device
				),
				model,
				scales=[1]
			)

	keypoints[:, 0] *= fact_i
	keypoints[:, 1] *= fact_j
	keypoints = keypoints[:, [1, 0, 2]]

	feat = {}
	feat['keypoints'] = keypoints
	feat['scores'] = scores
	feat['descriptors'] = descriptors

	return feat


def cv2D2netMatching(image1, image2, feat1, feat2, matcher="BF"):
	if(matcher == "BF"):

		t0 = time.time()
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		matches = bf.match(feat1['descriptors'], feat2['descriptors'])
		matches = sorted(matches, key=lambda x:x.distance)
		t1 = time.time()
		print("Time to extract matches: ", t1-t0)

		print("Number of raw matches:", len(matches))

		match1 = [m.queryIdx for m in matches]
		match2 = [m.trainIdx for m in matches]

		keypoints_left = feat1['keypoints'][match1, : 2]
		keypoints_right = feat2['keypoints'][match2, : 2]

		np.random.seed(0)

		t0 = time.time()
		model, inliers = ransac(
			(keypoints_left, keypoints_right),
			AffineTransform, min_samples=4,
			residual_threshold=8, max_trials=10000
		)
		t1 = time.time()
		print("Time for ransac: ", t1-t0)

		n_inliers = np.sum(inliers)
		print('Number of inliers: %d.' % n_inliers)

		inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
		inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
		placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
		image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)

		plt.figure(figsize=(20, 20))
		plt.imshow(image3)
		plt.axis('off')
		plt.show()

		return inlier_keypoints_left, inlier_keypoints_right, placeholder_matches


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


def siftMatching(frontImg, rearImg):
	frontImg = np.array(cv2.cvtColor(np.array(frontImg), cv2.COLOR_BGR2RGB))
	rearImg = np.array(cv2.cvtColor(np.array(rearImg), cv2.COLOR_BGR2RGB))

	surf = cv2.xfeatures2d.SURF_create(100)
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

	draw(kp1, kp2, good, frontImg, rearImg)

	n_inliers = len(good)

	return n_inliers, matches


def getTopImg(image, H, imgSize=400):
	warpImg = cv2.warpPerspective(image, H, (imgSize, imgSize))
	# cv2.imshow("Image", cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB))
	# cv2.waitKey(0)

	return warpImg


def orgKeypoints(keypoints1, keypoints2, matches, H1, H2):
	good = matches

	src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	src_pts = src_pts.squeeze(); dst_pts = dst_pts.squeeze()
	ones = np.ones((src_pts.shape[0], 1))

	src_pts = np.hstack((src_pts, ones))
	dst_pts = np.hstack((dst_pts, ones))

	orgSrc = np.linalg.inv(H1) @ src_pts.T
	orgDst = np.linalg.inv(H2) @ dst_pts.T

	orgSrc = orgSrc/orgSrc[2, :]
	orgDst = orgDst/orgDst[2, :]

	orgSrc = np.asarray(orgSrc)[0:2, :]
	orgDst = np.asarray(orgDst)[0:2, :]

	return orgSrc, orgDst


def drawOrg(image1, image2, orgSrc, orgDst):
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

	for i in range(orgSrc.shape[1]):
		im1 = cv2.circle(img1, (int(orgSrc[0, i]), int(orgSrc[1, i])), 3, (0, 0, 255), 1)
	for i in range(orgDst.shape[1]):
		im2 = cv2.circle(img2, (int(orgDst[0, i]), int(orgDst[1, i])), 3, (0, 0, 255), 1)

	# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)    
	# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)    

	im4 = cv2.hconcat([im1, im2])
	for i in range(orgSrc.shape[1]):
		im4 = cv2.line(im4, (int(orgSrc[0, i]), int(orgSrc[1, i])), (int(orgDst[0, i]) +  im1.shape[1], int(orgDst[1, i])), (0, 255, 0), 1)

	cv2.imshow("Image", im4)
	cv2.waitKey(0)


def getPerspKeypoints(rgbFile1, rgbFile2, HFile1, HFile2, model_file='models/d2_kinal_ipr.pth'):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")

	model = D2Net(
		model_file=model_file,
		use_relu=True,
		use_cuda=use_cuda
	)

	image1 = np.array(Image.open(rgbFile1))
	image2 = np.array(Image.open(rgbFile2))
	H1 = np.load(HFile1)
	H2 = np.load(HFile2)

	imgTop1 = getTopImg(image1, H1)
	imgTop2 = getTopImg(image2, H2)

	feat1 = extract(imgTop1, model, device)
	feat2 = extract(imgTop2, model, device)
	print("Features extracted.")


	keypoints1, keypoints2, matches = cv2D2netMatching(imgTop1, imgTop2, feat1, feat2, matcher="BF")

	orgSrc, orgDst = orgKeypoints(keypoints1, keypoints2, matches, H1, H2)
	drawOrg(image1, image2, orgSrc, orgDst)

	return orgSrc, orgDst


if __name__ == '__main__':
	WEIGHTS = 'models/d2_kinal_ipr.pth'

	srcR = argv[1] 
	trgR = argv[2]
	srcH = argv[3] 
	trgH = argv[4]

	orgSrc, orgDst = getPerspKeypoints(srcR, trgR, srcH, trgH, WEIGHTS)
	print(orgSrc); print(type(orgSrc))

	# np.savetxt('quantitative/src_pts.txt', orgSrc, delimiter=' ')
	# np.savetxt('quantitative/trg_pts.txt', orgDst, delimiter=' ')

	# siftMatching(image1, image2)