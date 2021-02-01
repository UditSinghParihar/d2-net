import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

import cv2
import matplotlib.pyplot as plt
import os
import re
from sys import exit
import time
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform


parser = argparse.ArgumentParser(description='Feature extraction script')
parser.add_argument('--dir1', type=str)
parser.add_argument('--dir2', type=str)

parser.add_argument(
	'--preprocessing', type=str, default='caffe',
	help='image preprocessing (caffe or torch)'
)

WEIGHTS = '/home/udit/d2-net/checkpoints/checkpoint_road_more/d2.15.pth'

parser.add_argument(
	'--model_file', type=str, default=WEIGHTS,
	help='path to the full model'
)
parser.add_argument(
	'--max_edge', type=int, default=1600,
	help='maximum image size at network input'
)
parser.add_argument(
	'--max_sum_edges', type=int, default=2800,
	help='maximum sum of image sizes at network input'
)

parser.add_argument(
	'--multiscale', dest='multiscale', action='store_true',
	help='extract multiscale features'
)
parser.set_defaults(multiscale=False)
parser.add_argument(
	'--no-relu', dest='use_relu', action='store_false',
	help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def extract(image, args, model, device):
	if len(image.shape) == 2:
		image = image[:, :, np.newaxis]
		image = np.repeat(image, 3, -1)

	resized_image = image
	if max(resized_image.shape) > args.max_edge:
		resized_image = scipy.misc.imresize(
			resized_image,
			args.max_edge / max(resized_image.shape)
		).astype('float')
	if sum(resized_image.shape[: 2]) > args.max_sum_edges:
		resized_image = scipy.misc.imresize(
			resized_image,
			args.max_sum_edges / sum(resized_image.shape[: 2])
		).astype('float')

	fact_i = image.shape[0] / resized_image.shape[0]
	fact_j = image.shape[1] / resized_image.shape[1]

	input_image = preprocess_image(
		resized_image,
		preprocessing=args.preprocessing
	)
	with torch.no_grad():
		if args.multiscale:
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
		# cv2.imshow('Matches', img3)
		# cv2.waitKey(0)

	else:
		print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		draw_params = dict(matchColor = (0,255,0),
						   singlePointColor = None,
						   matchesMask = matchesMask,
						   flags = 2)
		img3 = cv2.drawMatches(frontImg,kp1,rearImg,kp2,good,None,**draw_params)
		# cv2.imshow('Matches', img3)
		# cv2.waitKey(0)

	return img3


def drawMatches4(frontImg, rearImg):
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

	img3 = draw(kp1, kp2, good, frontImg, rearImg)

	n_inliers = len(good)

	return img3, n_inliers, matches


def	drawMatches1(image1, image2, feat1, feat2):
	matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
	print('Number of raw matches: %d.' % matches.shape[0])

	keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
	keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
	np.random.seed(0)
	model, inliers = ransac(
		(keypoints_left, keypoints_right),
		AffineTransform, min_samples=4,
		residual_threshold=8, max_trials=10000
	)
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


def drawMatches3(image1, image2, feat1, feat2):
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

	return image3


if __name__ == '__main__':
	outDir = '/scratch/udit/robotcar/overcast/ipm2/surf_400/'

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	args = parser.parse_args()

	frontImgs = natural_sort([file for file in os.listdir(args.dir1) if '.png' in file])
	rearImgs = natural_sort([file for file in os.listdir(args.dir2) if '.png' in file])

	model = D2Net(
		model_file=args.model_file,
		use_relu=args.use_relu,
		use_cuda=use_cuda
	)

	for i in range(len(frontImgs)):
		frontFile = os.path.join(args.dir1, frontImgs[i])
		rearFile = os.path.join(args.dir2, rearImgs[i])

		image1 = np.array(Image.open(frontFile).convert('L').resize((400, 400)))
		image1 = image1[:, :, np.newaxis]
		image1 = np.repeat(image1, 3, -1)
		image2 = np.array(Image.open(rearFile).convert('L').resize((400, 400)))
		image2 = image2[:, :, np.newaxis]
		image2 = np.repeat(image2, 3, -1)

		# cv2.imshow("Image", image1)
		# cv2.waitKey(0)
		# exit(1)

		t0 = time.time()
		feat1 = extract(image1, args, model, device)
		feat2 = extract(image2, args, model, device)
		t1 = time.time()
		print("Time for features extraction: ", t1-t0)
		# print("Features extracted.")
		
		# image3 = drawMatches3(image1, image2, feat1, feat2)
		image3, _ , _ = drawMatches4(image1, image2)

		outFile = os.path.join(outDir, str(i+1)+'.png')
		cv2.imwrite(outFile, image3)