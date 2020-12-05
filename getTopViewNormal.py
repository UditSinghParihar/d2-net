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


def getPointCloud(rgbFile, depthFile):
	thresh = 5.6

	depth = np.load(depthFile)
	rgb = Image.open(rgbFile)

	points = []
	colors = []

	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			
			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - centerX) * Z / focalLength
			Y = (v - centerY) * Z / focalLength
			
			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))

	points = np.asarray(points)
	colors = np.asarray(colors)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)

	downpcd = pcd.voxel_down_sample(voxel_size=0.03)
	
	return downpcd


def rotationMatrixFromVectors(vec1, vec2):
	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	rotationMatrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
	return rotationMatrix


def getNormals(pcd):
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	pcd.orient_normals_towards_camera_location()

	normals = np.asarray(pcd.normals)
	surfaceNormal = np.mean(normals, axis=0)
	
	return surfaceNormal


if __name__ == '__main__':
	rgbFile = argv[1]
	depthFile = argv[2]

	focalLength = 402.29
	centerX = 320.5
	centerY = 240.5
	scalingFactor = 1000.0

	pcd = getPointCloud(rgbFile, depthFile)

	surfaceNormal = getNormals(pcd)

	zAxis = np.array([0, 0, 1])
	rotationMatrix = rotationMatrixFromVectors(zAxis, -surfaceNormal)
	T = np.identity(4)
	T[0:3, 0:3] = rotationMatrix

	display(pcd)
	display(pcd, T)