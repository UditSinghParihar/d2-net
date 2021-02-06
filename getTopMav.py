import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt


def display(pcd, T=np.identity(4)):
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
	axis.transform(T)

	o3d.visualization.draw_geometries([pcd, axis])


def readDepth(path):
	# min_depth_percentile = 5
	# max_depth_percentile = 95
	min_depth_percentile = 1
	max_depth_percentile = 99

	with open(path, "rb") as fid:
		width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
												usecols=(0, 1, 2), dtype=int)
		fid.seek(0)
		num_delimiter = 0
		byte = fid.read(1)
		while True:
			if byte == b"&":
				num_delimiter += 1
				if num_delimiter >= 3:
					break
			byte = fid.read(1)
		array = np.fromfile(fid, np.float32)
	array = array.reshape((width, height, channels), order="F")

	depth_map = np.transpose(array, (1, 0, 2)).squeeze()

	min_depth, max_depth = np.percentile(depth_map, [min_depth_percentile, max_depth_percentile])
	print(min_depth, max_depth)

	depth_map[depth_map < min_depth] = min_depth
	depth_map[depth_map > max_depth] = max_depth
	# depth_map[depth_map < min_depth] = 0
	# depth_map[depth_map > max_depth] = 0

	return depth_map


def getPointCloud(rgbFile, depthFile):
	# thresh = 5.6
	thresh = 15.0

	depth = readDepth(depthFile)
	rgb = Image.open(rgbFile)

	# print(np.unique(depth))
	# print(rgb.size, depth.shape)
	# exit(1)

	points = []
	colors = []
	srcPxs = []

	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			
			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - centerX) * Z / focalX
			Y = (v - centerY) * Z / focalY
			
			srcPxs.append((u, v))
			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))

	srcPxs = np.asarray(srcPxs).T
	points = np.asarray(points)
	colors = np.asarray(colors)
	
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)

	# downpcd = pcd.voxel_down_sample(voxel_size=0.03)
	
	return pcd, srcPxs


if __name__ == '__main__':
	rgbFile = argv[1]
	depthFile = argv[2]

	# Medium colmap
	focalX = 743.738
	focalY = 744.136
	centerX = 800.0
	centerY = 399.5

	# High colmap
	# focalX = 1040.11
	# focalY = 1040.11
	# centerX = 1116.5
	# centerY = 559

	scalingFactor = 1.0

	pcd, srcPxs = getPointCloud(rgbFile, depthFile)
	display(pcd)

	# surfaceNormal = -getNormals(pcd)
	# surfaceNormal, planeDis = getPlane(pcd)

	# zAxis = np.array([0, 0, 1])
	# rotationMatrix = rotationMatrixFromVectors(zAxis, surfaceNormal)
	# T = np.identity(4)
	# T[0:3, 0:3] = rotationMatrix

	# # display(pcd)
	# # display(pcd, T)

	# # getImg(pcd, T)
	# getImgHomo(pcd, T, srcPxs, rgbFile)
