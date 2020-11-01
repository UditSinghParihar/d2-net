import numpy as np
from PIL import Image
import os
from sys import exit, argv
import csv
import torch
from torch.utils.data import Dataset
from lib.utils import preprocess_image
import cv2
from tqdm import tqdm

np.random.seed(0)


class PhotoTourism(Dataset):
	def __init__(self, rootDir, preprocessing):
		self.rootDir = rootDir
		self.preprocessing = preprocessing
		self.dataset = []

	def getImageFiles(self):
		imgFiles = os.listdir(self.rootDir)
		imgFiles = [os.path.join(self.rootDir, img) for img in imgFiles]

		return imgFiles

	def imgRot(self, img1):
		# img2 = img1.rotate(np.random.randint(low=0, high=360))
		img2 = img1.rotate(np.random.randint(low=0, high=2))

		return img2

	def imgCrop(self, img1, cropSize=256):
		w, h = img1.size
		left = np.random.randint(low = 0, high = w - (cropSize + 10))
		upper = np.random.randint(low = 0, high = h - (cropSize + 10))

		cropImg = img1.crop((left, upper, left+cropSize, upper+cropSize))
		
		# cropImg = cv2.cvtColor(np.array(cropImg), cv2.COLOR_BGR2RGB)
		# cv2.imshow("Image", cropImg)
		# cv2.waitKey(0)

		return cropImg

	def getCorr(self, img1, img2):
		im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
		im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
		
		surf = cv2.xfeatures2d.SURF_create(100)
		# surf = cv2.xfeatures2d.SIFT_create()

		kp1, des1 = surf.detectAndCompute(im1,None)
		kp2, des2 = surf.detectAndCompute(im2,None)

		if(len(kp1) < 128 or len(kp2) < 128):
			return [], []

		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		matches = bf.match(des1,des2)
		matches = sorted(matches, key=lambda x:x.distance)

		if(len(matches) > 800):
			matches = matches[0:800]
		elif(len(matches) < 128):
			return [], []

		pos1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
		pos2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T

		pos1[[0, 1]] = pos1[[1, 0]]
		pos2[[0, 1]] = pos2[[1, 0]]

		# for i in range(0, pos1.shape[1], 1):
		# 	im1 = cv2.circle(im1, (pos1[1, i], pos1[0, i]), 1, (0, 0, 255), 2)
		# for i in range(0, pos2.shape[1], 1):
		# 	im2 = cv2.circle(im2, (pos2[1, i], pos2[0, i]), 1, (0, 0, 255), 2)

		# im3 = cv2.hconcat([im1, im2])

		# for i in range(0, pos1.shape[1], 1):
		# 	im3 = cv2.line(im3, (int(pos1[1, i]), int(pos1[0, i])), (int(pos2[1, i]) +  im1.shape[1], int(pos2[0, i])), (0, 255, 0), 1)

		# im4 = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=2)
		# cv2.imshow('Image', im1)
		# cv2.imshow('Image2', im2)
		# cv2.imshow('Image3', im3)
		# cv2.imshow('Image4', im4)
		# cv2.waitKey(0)

		return pos1, pos2

	def build_dataset(self, cropSize=256):
		print("Building Dataset.")

		imgFiles = self.getImageFiles()

		for img in tqdm(imgFiles, total=len(imgFiles)):
			img1 = Image.open(img)

			if(img1.mode != 'RGB'):
				img1 = img1.convert('RGB')
			elif(img1.size[0] < cropSize or img1.size[1] < cropSize):
				continue

			img1 = self.imgCrop(img1, cropSize)
			img2 = self.imgRot(img1)

			img1 = np.array(img1)
			img2 = np.array(img2)

			pos1, pos2 = self.getCorr(img1, img2)
			if(len(pos1) == 0 or len(pos2) == 0):
				continue

			self.dataset.append((img1, img2, pos1, pos2))


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		image1, image2, pos1, pos2 = self.dataset[idx]

		image1 = preprocess_image(image1, preprocessing=self.preprocessing)
		image2 = preprocess_image(image2, preprocessing=self.preprocessing)

		return {
			'image1': torch.from_numpy(image1.astype(np.float32)),
			'image2': torch.from_numpy(image2.astype(np.float32)),
			'pos1': torch.from_numpy(pos1.astype(np.float32)),
			'pos2': torch.from_numpy(pos2.astype(np.float32))
		}


if __name__ == '__main__':
	rootDir = argv[1]

	training_dataset = PhotoTourism(rootDir, 'caffe')
	training_dataset.build_dataset()

	data = training_dataset[0]
	print(data['image1'].shape, data['image2'].shape, data['pos1'].shape, data['pos2'].shape)