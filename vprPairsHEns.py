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

			frontFeat = os.path.join(frontDir, os.path.basename(frontImg.replace('png', 'd2-net')))
			frontDict[os.path.basename(frontImg).replace('.png', '')] = np.load(frontFeat)

		rearFeats = natural_sort([file for file in os.listdir(rearDir) if '.d2-net' in file])
		rearFeatsFull = [os.path.join(rearDir, feat) for feat in rearFeats]
		
		for i in tqdm(range(len(rearFeats)), total=len(rearFeats)):
			rearDict[rearFeats[i].replace('.d2-net', '')] = np.load(rearFeatsFull[i])

		print("Features loaded.")

		return frontDict, rearDict

# Ensemble

def mnn_matcher_scorer(descriptors_a, descriptors_b, k=np.inf):
	device = descriptors_a.device
	sim = descriptors_a @ descriptors_b.t()
	val1, nn12 = torch.max(sim, dim=1)
	val2, nn21 = torch.max(sim, dim=0)
	ids1 = torch.arange(0, sim.shape[0], device=device)
	mask = (ids1 == nn21[nn12])
	matches = torch.stack([ids1[mask], nn12[mask]]).t()
	remaining_matches_dist = val1[mask]
	return matches, remaining_matches_dist


def numInliers(frontFeat1, rearFeat1, frontFeat2, rearFeat2):
	keypoints_a1 = frontFeat1['keypoints']
	descriptors_a1 = frontFeat1['descriptors']
	keypoints_a2 = frontFeat2['keypoints']
	descriptors_a2 = frontFeat2['descriptors']
	
	keypoints_b1 = rearFeat1['keypoints']
	descriptors_b1 = rearFeat1['descriptors']
	keypoints_b2 = rearFeat2['keypoints']
	descriptors_b2 = rearFeat2['descriptors']

	keypoints_a1 = keypoints_a1[:, [1, 0, 2]]
	keypoints_a2 = keypoints_a2[:, [1, 0, 2]]
	keypoints_b1 = keypoints_b1[:, [1, 0, 2]]
	keypoints_b2 = keypoints_b2[:, [1, 0, 2]]

	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda:1' if use_cuda else 'cpu')

	# calculating matches for both models
	matches1, dist_1 = mnn_matcher_scorer(
		torch.from_numpy(descriptors_a1).to(device=device),
		torch.from_numpy(descriptors_b1).to(device=device),
#                 len(matches1)
	)
	matches2, dist_2 = mnn_matcher_scorer(
		torch.from_numpy(descriptors_a2).to(device=device),
		torch.from_numpy(descriptors_b2).to(device=device),
#                 len(matches1)
	)

	full_matches = torch.cat([matches1, matches2])
	full_dist = torch.cat([dist_1, dist_2])
	assert len(full_dist)==(len(dist_1)+len(dist_2)), "something wrong"

	k_final = len(full_dist)//2
	# k_final = len(full_dist)
	# k_final = max(len(dist_1), len(dist_2))
	top_k_mask = torch.topk(full_dist, k=k_final)[1]
	first = []
	second = []

	for valid_id in top_k_mask:
		if valid_id<len(dist_1):
			first.append(valid_id)
		else:
			second.append(valid_id-len(dist_1))

	matches1 = matches1[torch.tensor(first, device=device).long()].data.cpu().numpy()
	matches2 = matches2[torch.tensor(second, device=device).long()].data.cpu().numpy()

	pos_a1 = keypoints_a1[matches1[:, 0], : 2]
	pos_b1 = keypoints_b1[matches1[:, 1], : 2]

	pos_a2 = keypoints_a2[matches2[:, 0], : 2]
	pos_b2 = keypoints_b2[matches2[:, 1], : 2]

	pos_a = np.concatenate([pos_a1, pos_a2], 0)
	pos_b = np.concatenate([pos_b1, pos_b2], 0)

	H, inliers = pydegensac.findHomography(pos_a, pos_b, 8.0, 0.99, 10000)

	n_inliers = np.sum(inliers)

	return n_inliers


def getPairs(queryPairs, frontDict1, rearDict1, frontDict2, rearDict2, rearDir1):
	
	matches = []

	rearImgs1 = natural_sort([file for file in os.listdir(rearDir1) if '.png' in file])
	rearImgs1 = [file.replace('.png', '') for file in rearImgs1]

	progressBar = tqdm(queryPairs, total=len(queryPairs))
	correct = 0.0
	total = 0.0
	for pair in progressBar:
		frontImg = pair[0]
		frontFeat1 = frontDict1[os.path.basename(frontImg).replace('.png', '')]
		frontFeat2 = frontDict2[os.path.basename(frontImg).replace('.png', '')]

		gtStId = int(os.path.basename(pair[1]).replace('.png', ''))
		gtEndId = int(os.path.basename(pair[2]).replace('.png', ''))

		dbStart = os.path.basename(pair[3]).replace('.png', '')
		dbEnd = os.path.basename(pair[4]).replace('.png', '')

		dbStartIdx = rearImgs1.index(dbStart)
		dbEndIdx = rearImgs1.index(dbEnd)

		maxInliers = -100
		maxRear = None

		# for idx in tqdm(range(dbStartIdx, dbEndIdx+1)):
		for idx in range(dbStartIdx, dbEndIdx+1):
			rearFeat1 = rearDict1[rearImgs1[idx]]
			rearFeat2 = rearDict2[rearImgs1[idx]]

			inliers = numInliers(frontFeat1, rearFeat1, frontFeat2, rearFeat2)
			
			if(maxInliers < inliers):
				maxInliers = inliers
				maxRear = rearImgs1[idx]

		matches.append([frontImg, maxRear, str(maxInliers)])
		total += 1.0
		
		if(gtStId < int(maxRear) < gtEndId):
			correct += 1.0
		progressBar.set_postfix(retrieve=('{}/{}({:.2f})'.format(correct, total, correct*100.0/total)))

	return matches


def writeMatches(matches):
	with open('dataGenerate/vprOutputHDiffEnsemble.csv', 'w', newline='') as file:
		writer = csv.writer(file)

		title = ['FrontImage', 'RearImage', 'Correspondences']
		writer.writerow(title)

		for match in matches:
			writer.writerow(match)


if __name__ == '__main__':
	pairsFile = argv[1]
	frontDir1 = argv[2]
	rearDir1 = argv[3]
	frontDir2 = argv[4]
	rearDir2 = argv[5]

	queryPairs = readQuery(pairsFile)

	frontDict1, rearDict1 = loadFeat(queryPairs, frontDir1, rearDir1)
	frontDict2, rearDict2 = loadFeat(queryPairs, frontDir2, rearDir2)

	matches = getPairs(queryPairs, frontDict1, rearDict1, frontDict2, rearDict2, rearDir1)

	writeMatches(matches)
