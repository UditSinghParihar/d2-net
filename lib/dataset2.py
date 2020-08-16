import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from sys import exit, argv
import csv
import torch
from torch.utils.data import Dataset
from lib.utils import preprocess_image


class LabDataset(Dataset):
	def __init__(self, rootDir, imgPairs, poses, intrinsics, preprocessing):
		self.rootDir = rootDir
		self.imgPairs = imgPairs
		self.poses = poses
		self.intrinsics = intrinsics
		self.preprocessing = preprocessing
		

	def getImageFiles(self):
		imgFiles = []
		
		with open(self.imgPairs) as csvFile:
			csvReader = csv.reader(csvFile, delimiter=',')

			for i, row in enumerate(csvReader):
				if(i == 0):
					continue
				else:
					imgFiles.append(row)

		return imgFiles


	def getPoses(self):
		poses2 = np.load(self.poses)

		poses1 = np.zeros((poses2.shape[0], 4, 4))
		poses1[:, 0:4, 0:4] = np.identity(4)

		return (poses1, poses2)


	def getIntrinsics(self):
		K = np.load(self.intrinsics)

		return K


	def build_dataset(self):
		print("Building Dataset.")

		self.dataset = []

		imgFiles = self.getImageFiles()
		poses1, poses2 = self.getPoses()
		K = self.getIntrinsics()
		bbox = np.array([0.0, 0.0])

		for i in range(len(imgFiles)):
			rgbFile1, depthFile1, rgbFile2, depthFile2 = imgFiles[i]
			
			rgbFile1 = os.path.join(self.rootDir, rgbFile1)
			depthFile1 = os.path.join(self.rootDir, depthFile1)
			rgbFile2 = os.path.join(self.rootDir, rgbFile2)
			depthFile2 = os.path.join(self.rootDir, depthFile2)

			rgb1 = Image.open(rgbFile1)
			depth1 = Image.open(depthFile1)
			rgb2 = Image.open(rgbFile2)
			depth2 = Image.open(depthFile2)			

			if(depth1.mode != "I" or depth2.mode != "I"):
				raise Exception("Depth image is not in intensity format")
			
			if(rgb1.mode != 'RGB'):
				rgb1 = rgb1.convert('RGB')
			
			if(rgb2.mode != 'RGB'):
				rgb2 = rgb2.convert('RGB')

			rgb1 = np.array(rgb1)
			rgb2 = np.array(rgb2)
			depth1 = np.array(depth1)/1000.0
			depth2 = np.array(depth2)/1000.0

			assert(rgb1.shape[0] == depth1.shape[0] and rgb1.shape[1] == depth1.shape[1])
			assert(rgb2.shape[0] == depth2.shape[0] and rgb2.shape[1] == depth2.shape[1])

			pose1 = poses1[i]
			pose2 = poses2[i]

			self.dataset.append((
				rgb1,
				depth1,
				K,
				pose1,
				bbox,
				rgb2,
				depth2,
				K,
				pose2,
				bbox
			))


	def __len__(self):
		return len(self.dataset)


	def __getitem__(self, idx):
		image1, depth1, intrinsics1, pose1, bbox1, image2, depth2, intrinsics2, pose2, bbox2 = self.dataset[idx]

		image1 = preprocess_image(image1, preprocessing=self.preprocessing)
		image2 = preprocess_image(image2, preprocessing=self.preprocessing)

		return {
			'image1': torch.from_numpy(image1.astype(np.float32)),
			'depth1': torch.from_numpy(depth1.astype(np.float32)),
			'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
			'pose1': torch.from_numpy(pose1.astype(np.float32)),
			'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
			'image2': torch.from_numpy(image2.astype(np.float32)),
			'depth2': torch.from_numpy(depth2.astype(np.float32)),
			'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
			'pose2': torch.from_numpy(pose2.astype(np.float32)),
			'bbox2': torch.from_numpy(bbox2.astype(np.float32))
		}

	 
if __name__ == '__main__':
	rootDir = "/scratch/udit/"

	imgPairs = argv[1]
	poses = argv[2]
	intrinsics = argv[3]

	training_dataset = LabDataset(rootDir, imgPairs, poses, intrinsics)

	training_dataset.build_dataset()