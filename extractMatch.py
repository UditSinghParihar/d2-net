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
from sys import exit
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform


parser = argparse.ArgumentParser(description='Feature extraction script')
parser.add_argument('imgs', type=str, nargs=2)
parser.add_argument(
	'--preprocessing', type=str, default='caffe',
	help='image preprocessing (caffe or torch)'
)
# parser.add_argument(
# 	'--model_file', type=str, default='models/d2_tf.pth',
# 	help='path to the full model'
# )

# WEIGHTS = 'models/d2_tf.pth'
# WEIGHTS = '/home/dhagash/d2-net/d2-net_udit/checkpoints/checkpoint_PT_highRot_epoch/d2.15.pth'
# WEIGHTS = 'results/train_corr14_360/checkpoints/d2.10.pth'
# WEIGHTS = 'results/train_corr15_gazebo/checkpoints/d2.10.pth'
# WEIGHTS = 'results/train_corr16_gz_15ep_curriculum/checkpoints/d2.15.pth'
# WEIGHTS = '/home/udit/d2-net/checkpoints/checkpoint_rcar_allRoad/d2.10.pth'
# WEIGHTS = '/home/udit/d2-net/checkpoints/checkpoint_rcar_crop/d2.10.pth'
WEIGHTS = '/home/udit/udit/d2-net/results/train_corr18_stability_term/checkpoints/d2.09.pth'

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
	'--output_extension', type=str, default='.d2-net',
	help='extension for the output'
)
parser.add_argument(
	'--output_type', type=str, default='npz',
	help='output file type (npz or mat)'
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


def extract(image, args, model, device):
# def extract(file, args, model, device):
# 	image = imageio.imread(file)
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


def	drawMatches(image1, image2, feat1, feat2):
# def	drawMatches(file1, file2, feat1, feat2):
	# image1 = np.array(Image.open(file1))
	# image2 = np.array(Image.open(file2))

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


def	drawMatches2(image1, image2, feat1, feat2):
	# image1 = cv2.imread(file1)
	# image2 = cv2.imread(file2)
	image1 = np.array(cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2RGB))
	image2 = np.array(cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2RGB))

	matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
	keypoints_left = feat1['keypoints'][matches[:, 0], : 2].T
	keypoints_right = feat2['keypoints'][matches[:, 1], : 2].T

	# print(keypoints_left.shape, keypoints_right.shape)

	for i in range(keypoints_left.shape[1]):
		image1 = cv2.circle(image1, (int(keypoints_left[0, i]), int(keypoints_left[1, i])), 2, (0, 0, 255), 4)
	for i in range(keypoints_right.shape[1]):
		image2 = cv2.circle(image2, (int(keypoints_right[0, i]), int(keypoints_right[1, i])), 2, (0, 0, 255), 4)

	im4 = cv2.hconcat([image1, image2])	

	for i in range(keypoints_left.shape[1]):
		im4 = cv2.line(im4, (int(keypoints_left[0, i]), int(keypoints_left[1, i])), (int(keypoints_right[0, i]) +  image1.shape[1], int(keypoints_right[1, i])), (0, 255, 0), 1)

	cv2.imshow("Image_lines", im4)
	cv2.waitKey(0)


if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	args = parser.parse_args()

	model = D2Net(
		model_file=args.model_file,
		use_relu=args.use_relu,
		use_cuda=use_cuda
	)

	# image1 = np.array(Image.open(args.imgs[0]))
	# image2 = np.array(Image.open(args.imgs[1]))

	image1 = np.array(Image.open(args.imgs[0]).convert('L').resize((500, 500)))
	image2 = np.array(Image.open(args.imgs[1]).convert('L').resize((500, 500)))

	image1 = image1[:, :, np.newaxis]
	image1 = np.repeat(image1, 3, -1)

	image2 = image2[:, :, np.newaxis]
	image2 = np.repeat(image2, 3, -1)

	# cv2.imshow("Image", image1)
	# cv2.waitKey(0)
	# exit(1)

	feat1 = extract(image1, args, model, device)
	feat2 = extract(image2, args, model, device)
	print("Features extracted.")

	drawMatches(image1, image2, feat1, feat2)
	# drawMatches2(image1, image2, feat1, feat2)


	# feat1 = extract(args.imgs[0], args, model, device)
	# feat2 = extract(args.imgs[1], args, model, device)
	# print("Features extracted.")

	# drawMatches(args.imgs[0], args.imgs[1], feat1, feat2)

	# drawMatches2(args.imgs[0], args.imgs[1], feat1, feat2)