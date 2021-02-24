import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import numpy as np
import copy
import os
import re
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial.transform import Rotation as R


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def readPoses(file):
	f = open(file, 'r')
	A = f.readlines()
	f.close()

	poses = []

	for i, line in enumerate(A):
		T = np.identity(4)
		row = line.split(' ')
		px, py, pz, qx, qy, qz, qw = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
		
		Rot = R.from_quat([qx, qy, qz, qw])
		T[0:3, 0:3] = Rot.as_dcm()
		T[0, 3] = px
		T[1, 3] = py
		T[2, 3] = pz

		poses.append(T)

	return poses


def display(pcd, T=np.identity(4)):
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
	axis.transform(T)

	o3d.visualization.draw_geometries([pcd, axis])


def draw_registration_result(source, target, transformation):
	geometries = []

	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	# source_temp.paint_uniform_color([1, 0.706, 0])
	# target_temp.paint_uniform_color([0, 0.651, 0.929])

	# X_target = Ttarget_source @ X_source
	# Ttarget_source : source wrt target
	source_temp.transform(transformation)
	geometries.append(source_temp); geometries.append(target_temp)
	
	axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2.transform(transformation)
	geometries.append(axis1); geometries.append(axis2)
	
	o3d.visualization.draw_geometries(geometries)


def readDepth(depthFile):
	depth = Image.open(depthFile)
	if depth.mode != "I":
		raise Exception("Depth image is not in intensity format")

	return np.asarray(depth)


def getPointCloud(rgbFile, depthFile):
	# pts = [(275, 462), (435, 275), (465, 27), (41, 197)]
	# poly = Polygon(pts)

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
			
			# p = Point(u, v)
			# if not (p.within(poly)):
			# 	continue

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
	
	return pcd, srcPxs


def right2left():
	# Right wrt left = TL_R = TR2L

	thetaX = math.radians(90)
	thetaY = math.radians(-90)

	Rx = np.array([[1, 0, 0], [0, math.cos(thetaX), -math.sin(thetaX)], [0, math.sin(thetaX), math.cos(thetaX)]])
	Ry = np.array([[math.cos(thetaY), 0, math.sin(thetaY)], [0, 1, 0], [-math.sin(thetaY), 0, math.cos(thetaY)]])

	TR2L = np.identity(4)
	TR2L[0:3, 0:3] =  Ry @ Rx 
	
	return TR2L


def getRelativeT(poses, srcIdx, trgIdx):
	# Right hand : ROS - X Forward, Y Left, Z Upward
	# Left hand : General CV algorithms - Z Forward, X Right, Y Downwards
	# Convert left PC to right, apply right coordinate transform, convert right PC to left

	Ttrg = poses[trgIdx]
	Tsrc = poses[srcIdx]

	Ttrg_srcR = np.linalg.inv(Ttrg) @ Tsrc
	TR2L = right2left()
	Ttrg_srcL = TR2L @ Ttrg_srcR @ np.linalg.inv(TR2L)
	
	return Ttrg_srcL


def getHomo(srcPcd, Ttrg_src, srcPxs):
	srcPcd.transform(Ttrg_src)



if __name__ == '__main__':
	gtPoses = argv[1]
	rgbDir = argv[2]
	depthDir = argv[3]

	rgbImgs = natural_sort(os.listdir(rgbDir))
	depthImgs = natural_sort(os.listdir(depthDir))
	poses = readPoses(gtPoses)

	rgbImgs = [os.path.join(rgbDir, img) for img in rgbImgs if ".jpg" in img]
	depthImgs = [os.path.join(depthDir, img) for img in depthImgs if ".png" in img]

	srcIdx = 0
	trgIdx = 1399

	# Realsense D455
	focalX = 382.1996765136719
	focalY = 381.8395690917969
	centerX = 312.7102355957031
	centerY = 247.72047424316406

	scalingFactor = 1000.0

	# Source wrt target in Left hand
	Ttrg_src = getRelativeT(poses, srcIdx, trgIdx)

	srcPcd, srcPxs = getPointCloud(rgbImgs[srcIdx], depthImgs[srcIdx])
	trgPcd, trgPxs = getPointCloud(rgbImgs[trgIdx], depthImgs[trgIdx])

        # display(srcPcd)
	# display(trgPcd)
	# draw_registration_result(source=srcPcd, target=trgPcd, transformation=Ttrg_src)

	print(Ttrg_src)

	# getHomo(srcPcd, Ttrg_src, srcPxs)


