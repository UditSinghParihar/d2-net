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
parser.add_argument('dir', type=str)
parser.add_argument(
	'--preprocessing', type=str, default='caffe',
	help='image preprocessing (caffe or torch)'
)

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


if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	args = parser.parse_args()

	model = D2Net(
		model_file=args.model_file,
		use_relu=args.use_relu,
		use_cuda=use_cuda
	)

	imgFiles = os.listdir(args.dir)
	imgFiles = [os.path.join(args.dir, img) for img in imgFiles]

	for imgFile in tqdm(imgFiles, total=len(imgFiles)):
		image = np.array(Image.open(imgFile).convert('L'))
		image = image[:, :, np.newaxis]
		image = np.repeat(image, 3, -1)

		feat = extract(image, args, model, device)

		path = imgFile.replace('.png', '') + args.output_extension
		
		with open(path, 'wb') as output_file:
			np.savez(
				output_file,
				keypoints=feat['keypoints'],
				scores=feat['scores'],
				descriptors=feat['descriptors']
			)

		# fig, ax = plt.subplots(1, 1)
		# ax.imshow(image)
		# ax.set_title('Front View')
		# plt.show()