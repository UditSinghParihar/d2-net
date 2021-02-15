import numpy as np
import copy
import open3d as o3d
from sys import argv
from PIL import Image
import math
from extractMatchTop import getPerspKeypoints


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
			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - centerX) * Z / focalX
			Y = (v - centerY) * Z / focalY
			
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


def convertPts(A):
	X = A[0]; Y = A[1] 

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


if __name__ == "__main__":
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
	srcH = argv[5]
	trgH = argv[6]

	srcPts, trgPts = getPerspKeypoints(srcR, trgR, srcH, trgH)
	# exit(1)
	
	srcPts = convertPts(srcPts)
	trgPts = convertPts(trgPts)

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
	print(trans_init)

	draw_registration_result(srcCld, trgCld, trans_init)
