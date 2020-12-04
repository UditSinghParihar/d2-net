import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt


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
			
			# Xtemp = X; Ytemp = Y; Ztemp = Z
			# X = Ztemp; Y = -Xtemp; Z = -Ytemp

			# if(Z < 0): continue
			
			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))

	points = np.asarray(points)
	colors = np.asarray(colors)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)

	downpcd = pcd.voxel_down_sample(voxel_size=0.01)
	# points = np.asarray(downpcd.points)
	
	# ones = np.ones((points.shape[0], 1))
	# points = np.hstack((points, ones))


	# points = T @ points.T

	# downpcd.points = o3d.utility.Vector3dVector((points.T)[:, 0:3])

	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
	o3d.visualization.draw_geometries([pcd, axis])
	
	return pcd


def rotation_matrix_from_vectors(vec1, vec2):
	""" Find the rotation matrix that aligns vec1 to vec2
	:param vec1: A 3d "source" vector
	:param vec2: A 3d "destination" vector
	:return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
	"""
	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
	return rotation_matrix


# def getNormals(pcd):
# 	pcd.estimate_normals(pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# 	o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
	rgbFile = argv[1]
	depthFile = argv[2]

	focalLength = 402.29
	centerX = 320.5
	centerY = 240.5
	scalingFactor = 1000.0

	pcd = getPointCloud(rgbFile, depthFile)

	# getNormals(pcd)