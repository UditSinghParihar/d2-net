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
from sys import exit, argv
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform



def numInliers(feat1, feat2):
	matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)

	keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
	keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
	np.random.seed(0)
	model, inliers = ransac(
		(keypoints_left, keypoints_right),
		AffineTransform, min_samples=4,
		residual_threshold=8, max_trials=10000
	)
	n_inliers = np.sum(inliers)

	return n_inliers



if __name__ == '__main__':
	frontDir = argv[1]
	rearDir = argv[2]

	files = os.listdir(frontDir)
	featFrontFiles = [os.path.join(frontDir, file) for file in files if '.d2-net' in file]

	files = os.listdir(rearDir)
	featRearFiles = [os.path.join(rearDir, file) for file in files if '.d2-net' in file]

	featFront = [np.load(featFile) for featFile in featFrontFiles] 		
	featRear = [np.load(featFile) for featFile in featRearFiles]

	print("Features extracted")

	pairs = []

	for i, ftRear in enumerate(featRear):
		maxInliers = -100
		idx = -1

		for j, ftFront in enumerate(featFront):
			inliers = numInliers(ftRear, ftFront)
			print(featRearFiles[i], featFrontFiles[j], inliers)
			if(maxInliers < inliers):
				maxInliers = inliers
				idx = j

		pair = (featRearFiles[i], featFrontFiles[idx], maxInliers)
		print(pair)
		pairs.append(pair)

		exit(1)