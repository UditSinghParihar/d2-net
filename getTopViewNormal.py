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

	# downpcd = pcd.voxel_down_sample(voxel_size=0.03)
	
	return pcd


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


def getPointsInCamera(pcd, T):
	pcd.transform(np.linalg.inv(T))

	return pcd


def extractPCD(pcd):
	pcdPoints = np.asarray(pcd.points)
	pcdColor = np.asarray(pcd.colors).T

	return pcdPoints, pcdColor


def getPixels(pcdPoints):
	K = np.array([[focalLength, 0, centerX], [0, focalLength, centerY], [0, 0, 1]])
	pxh = K @ pcdPoints.T

	pxh[0, :] = pxh[0, :]/pxh[2, :]
	pxh[1, :] = pxh[1, :]/pxh[2, :]
	pxh[2, :] = pxh[2, :]/pxh[2, :]

	return pxh[0:2, :]


def resizePxs(pxs, imgSize):
	minX = np.min(pxs[0, :])
	minY = np.min(pxs[1, :])

	if(minX < 0):
		pxs[0, :] += (np.abs(minX) + 2)
	if(minY < 0):
		pxs[1, :] += (np.abs(minY) + 2)

	maxX = np.max(pxs[0, :])
	maxY = np.max(pxs[1, :])

	ratioX = imgSize/maxX
	ratioY = imgSize/maxY

	pxs[0, :] *= ratioX
	pxs[1, :] *= ratioY

	return pxs


def pxsToImg(pxs, pcdColor, imgSize):
	height = imgSize; width = imgSize

	img = np.zeros((height, width, 3), np.uint8)

	for i in range(pxs.shape[1]):
		r = int(pxs[1, i]); c = int(pxs[0, i])
		if(r<height and c<width and r>0 and c>0):
			red = 255*pcdColor[0, i]; green = 255*pcdColor[1, i]; blue = 255*pcdColor[2, i]
			img[r, c] = (blue, green, red)
			
	return img


def getImg(pcd, T):
	pcd = getPointsInCamera(pcd, T)

	pcdPoints, pcdColor = extractPCD(pcd)

	pxs = getPixels(pcdPoints)

	imgSize = 400
	pxs = resizePxs(pxs, imgSize)

	img = pxsToImg(pxs, pcdColor, imgSize)

	cv2.imshow("image", img)
	cv2.waitKey(0)


def get3Dpoints(rgbFile, surfaceNormal, d=10):
	rgb = Image.open(rgbFile)

	K = np.array([[focalLength, 0, centerX], [0, focalLength, centerY], [0, 0, 1]])

	n1, n2, n3 = surfaceNormal

	points = []
	colors = []

	for v in range(rgb.size[1]):
		for u in range(rgb.size[0]):
			colors.append(rgb.getpixel((u, v)))
			
			x = np.array([u, v, 1]).reshape(3, 1)
			ray = np.linalg.inv(K) @ x
			l, m , n = ray.reshape(3)

			# x1, y1, z1 = v, u, 1			
			x1, y1, z1 = u, v, 1

			t = -(n1*x1 + n2*y1 + n3*z1 + d)/(n1*l + n2*m + n3*n)

			X = l*t + x1
			Y = m*t + y1
			Z = n*t + z1

			points.append((X, Y, Z))

			# exit(1)

	points = np.asarray(points)
	colors = np.asarray(colors)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)

	display(pcd)

	# print(surfaceNormal)



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

	# display(pcd)
	# display(pcd, T)

	# getImg(pcd, T)

	get3Dpoints(rgbFile, surfaceNormal, d=10)