import numpy as np
import copy
import open3d as o3d
from sys import argv
from PIL import Image
import math
from squaternion import quat2euler, Quaternion
import pyquaternion


def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	# source_temp.paint_uniform_color([1, 0.706, 0])
	# target_temp.paint_uniform_color([0, 0.651, 0.929])
	source_temp.transform(transformation)
	trgSph.append(source_temp); trgSph.append(target_temp)
	axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2.transform(transformation)
	trgSph.append(axis1); trgSph.append(axis2)
	o3d.visualization.draw_geometries(trgSph)


def draw_registration_result2(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	# source_temp.paint_uniform_color([1, 0.706, 0])
	# target_temp.paint_uniform_color([0, 0.651, 0.929])
	source_temp.transform(transformation)
	o3d.visualization.draw_geometries([source_temp, target_temp])


def readDepth(depthFile):
	depth = Image.open(depthFile)
	if depth.mode != "I":
		raise Exception("Depth image is not in intensity format")

	return np.asarray(depth)


def getPointCloud2(rgbFile, depthFile, pts):
	# thresh = 5.6

	# depth = Image.open(depthFile)
	# if depth.mode != "I":
	# 	raise Exception("Depth image is not in intensity format")

	# rgb = Image.open(rgbFile)

	thresh = 15.0

	depth = readDepth(depthFile)
	rgb = Image.open(rgbFile)

	points = []
	colors = []

	corIdx = [-1]*len(pts)
	corPts = [None]*len(pts)
	ptIdx = 0

	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			# Z = depth.getpixel((u,v)) / scalingFactor
			# if Z==0: continue
			# if (Z > thresh): continue

			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - centerX) * Z / focalX
			Y = (v - centerY) * Z / focalY
			
			# Xtemp = X; Ytemp = Y; Ztemp = Z
			# X = Ztemp; Y = -Xtemp; Z = -Ytemp

			# if(Z < 0): continue
			
			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))

			if((u, v) in pts):
				# print("Point found.")
				index = pts.index((u, v))
				corIdx[index] = ptIdx
				corPts[index] = (X, Y, Z)

			ptIdx = ptIdx+1

	points = np.asarray(points)
	colors = np.asarray(colors)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)
	
	return pcd, corIdx, corPts


def readCorr(file):
	f = open(file, 'r')
	A = f.readlines()
	f.close()

	X = A[0].split(' '); Y = A[1].split(' ') 
	x = [];	y = []

	for i in range(len(X)):
		x.append(int(float(X[i])))		

	for i in range(len(Y)):
		y.append(int(float(Y[i])))		
	
	pts = []
	for i in range(len(x)):
		pts.append((x[i], y[i]))

	return pts


def getSphere(pts):
	sphs = []

	for ele in pts:
		if(ele is not None):
			sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
			sphere.paint_uniform_color([0.9, 0.2, 0])

			trans = np.identity(4)
			trans[0, 3] = ele[0]
			trans[1, 3] = ele[1]
			trans[2, 3] = ele[2]

			sphere.transform(trans)
			sphs.append(sphere)

	return sphs


def get3dCor(src, trg):
	corr = []

	for sId, tId in zip(src, trg): 
		if(sId != -1 and tId != -1):
			corr.append((sId, tId))

	corr = np.asarray(corr)
	
	return corr


def rot2euler(R):
	quat = pyquaternion.Quaternion(matrix=R)
	quat2 = Quaternion(quat.elements[0], quat.elements[1], quat.elements[2], quat.elements[3])
	euler = quat2euler(*quat2, degrees=True)

	return euler


if __name__ == "__main__":
	# threshold = 0.02

	# focalLength = 617.19
	# centerX = 314.647
	# centerY = 246.577
	# scalingFactor = 1000.0

	# Realsense D455
	focalX = 382.1996765136719
	focalY = 381.8395690917969
	centerX = 312.7102355957031
	centerY = 247.72047424316406

	scalingFactor = 1000.0

	srcR = argv[1]
	srcD = argv[2]
	trgR = argv[3]
	trgD = argv[4]
	srcPts = argv[5]
	trgPts = argv[6]

	srcPts = readCorr(srcPts)
	trgPts = readCorr(trgPts)

	srcCld, srcIdx, srcCor = getPointCloud2(srcR, srcD, srcPts)
	trgCld, trgIdx, trgCor = getPointCloud2(trgR, trgD, trgPts)

	srcSph = getSphere(srcCor)
	trgSph = getSphere(trgCor)	
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	srcSph.append(srcCld); srcSph.append(axis)
	trgSph.append(trgCld); trgSph.append(axis)
	o3d.visualization.draw_geometries(srcSph)
	o3d.visualization.draw_geometries(trgSph)
	# exit(1)

	corr = get3dCor(srcIdx, trgIdx)

	p2p = o3d.registration.TransformationEstimationPointToPoint()
	trans_init = p2p.compute_transformation(srcCld, trgCld, o3d.utility.Vector2iVector(corr))
	# euler = rot2euler(trans_init[0:3, 0:3])
	# print(trans_init[0, 3], trans_init[1, 3], euler[2])
	print(trans_init)

	draw_registration_result(srcCld, trgCld, trans_init)
