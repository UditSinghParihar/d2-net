import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


def display(pcd, T=np.identity(4)):
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
	axis.transform(T)

	o3d.visualization.draw_geometries([pcd, axis])


def readDepth(depthFile):
	depth = Image.open(depthFile)
	if depth.mode != "I":
		raise Exception("Depth image is not in intensity format")

	return np.asarray(depth)


def getPointCloud(rgbFile, depthFile):
	# 100
	# pts = [(292, 342), (514, 465), (506, 184), (247, 186)]
	# 1000
	pts = [(310, 461), (402, 222), (401, 37), (217, 125)]
	# 905
	# pts = [(44, 394), (353, 392), (541, 322), (316, 160)]
	poly = Polygon(pts)

	# thresh = 5.6
	thresh = 15.0

	depth = readDepth(depthFile)
	rgb = Image.open(rgbFile)

	# cv2.imshow("Image",  np.array(cv2.cvtColor(np.array(rgb), cv2.COLOR_BGR2RGB)))
	# cv2.waitKey(0)

	# print(type(depth))
	# print(np.unique(depth))
	# print(rgb.size, depth.shape)
	# exit(1)

	points = []
	colors = []
	srcPxs = []

	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			
			p = Point(u, v)
			if not (p.within(poly)):
				continue

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


def rotationMatrixFromVectors(vec1, vec2):
	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	rotationMatrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
	return rotationMatrix


def getNormals(pcd):
	pcdNormal = copy.deepcopy(pcd)
	pcdNormal.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	pcdNormal.orient_normals_towards_camera_location()

	normals = np.asarray(pcdNormal.normals)
	surfaceNormal = np.mean(normals, axis=0)
	
	return surfaceNormal


def getPointsInCamera(pcd, T):
	pcd.transform(np.linalg.inv(T))

	mean = np.mean(np.asarray(pcd.points), axis=0)
	TCent = np.identity(4)
	TCent[0, 3] = mean[0]
	TCent[1, 3] = mean[1]
	display(pcd, TCent)
	pcd.transform(np.linalg.inv(TCent))

	return pcd


def extractPCD(pcd):
	pcdPoints = np.asarray(pcd.points)
	pcdColor = np.asarray(pcd.colors).T

	return pcdPoints, pcdColor


def getPixels(pcdPoints):
	K = np.array([[focalX, 0, centerX], [0, focalY, centerY], [0, 0, 1]])
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


def getImgHomo(pcd, T, srcPxs, rgbFile):
	pcd = getPointsInCamera(pcd, T)

	pcdPoints, pcdColor = extractPCD(pcd)
	# print(np.mean(pcdPoints, axis=0))

	trgPxs = getPixels(pcdPoints)

	imgSize = 400
	trgPxs = resizePxs(trgPxs, imgSize)

	homographyMat, status = cv2.findHomography(srcPxs.T, trgPxs.T)
	orgImg = cv2.cvtColor(np.array(Image.open(rgbFile)), cv2.COLOR_BGR2RGB)
	warpImg = cv2.warpPerspective(orgImg, homographyMat, (imgSize, imgSize))

	cv2.imshow("Image", warpImg)
	cv2.waitKey(0)

	return warpImg, homographyMat


def getPlane(pcd):
	planeModel, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
	[a, b, c, d] = planeModel
	surfaceNormal = [-a, -b, -c]
	# surfaceNormal = [a, b, c]
	planeDis = d
	# print("Plane equation: {}x + {}y + {}z + {} = 0".format(a, b, c, d))
	return surfaceNormal, planeDis


def getTopImage(rgbFile, depthFile):
	pcd, srcPxs = getPointCloud(rgbFile, depthFile)

	surfaceNormal = -getNormals(pcd)
	# surfaceNormal, planeDis = getPlane(pcd)
	# display(pcd); exit(1)

	zAxis = np.array([0, 0, 1])
	rotationMatrix = rotationMatrixFromVectors(zAxis, surfaceNormal)
	T = np.identity(4)
	T[0:3, 0:3] = rotationMatrix

	# display(pcd)
	# display(pcd, T)

	warpImg, homographyMat = getImgHomo(pcd, T, srcPxs, rgbFile)
	
	return warpImg, homographyMat


if __name__ == '__main__':
	rgbFile = argv[1]
	depthFile = argv[2]

	# Realsense D415
	# focalX = 607.8118896484375
	# focalY = 606.7265625
	# centerX = 324.09228515625
	# centerY = 235.1124725341797

	# Realsense D455
	focalX = 382.1996765136719
	focalY = 381.8395690917969
	centerX = 312.7102355957031
	centerY = 247.72047424316406

	scalingFactor = 1000.0

	warpImg, homographyMat = getTopImage(rgbFile, depthFile)

	cv2.imwrite('top1000img.png', warpImg)
	np.save('top1000.npy', homographyMat)