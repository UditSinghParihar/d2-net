import numpy as np
import copy
import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import cv2
import os
from scipy.spatial.transform import Rotation as R
import re
import time


def warp(u1, v1, Z1, K, T2_1):
	X1 = (u1 - K[0, 2]) * (Z1 / K[0, 0])
	Y1 = (v1 - K[1, 2]) * (Z1 / K[1, 1])

	XYZ1_hom = np.vstack((
		X1.reshape(1, -1),
		Y1.reshape(1, -1),
		Z1.reshape(1, -1),
		np.ones((1, Z1.size))
		))

	XYZ2_hom = T2_1 @ XYZ1_hom
	XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].reshape(1, -1)

	uv2_hom = K @ XYZ2
	uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].reshape(1, -1)

	return uv2[0, :].reshape(1, -1), uv2[1, :].reshape(1, -1)


def getSrcPx(imgFile, HFile, imgSize=400, draw=False):
	H = np.load(HFile)
	img = cv2.cvtColor(np.array(Image.open(imgFile)), cv2.COLOR_BGR2RGB)

	warpImg = cv2.warpPerspective(img, H, (imgSize, imgSize))

	surf = cv2.xfeatures2d.SURF_create(5)
	# surf = cv2.xfeatures2d.SIFT_create()
	kp, des = surf.detectAndCompute(warpImg, None)

	kpWarp = np.float32([px.pt for px in kp])
	
	ones = np.ones((kpWarp.shape[0], 1))
	kpOrg = np.linalg.inv(H) @ np.hstack((kpWarp, ones)).T
	kpOrg = kpOrg[0:2, :]/kpOrg[2, :]

	if(draw == True):
		for i in range(kpOrg.shape[1]):
			im1 = cv2.circle(img, (int(kpOrg[0, i]), int(kpOrg[1, i])), 3, (0, 0, 255), 1)
		cv2.imshow('Plotted', im1)
		cv2.imshow('warped', warpImg)
		cv2.waitKey(0)

	return kpOrg[0, :].reshape(1, -1).astype(int), kpOrg[1, :].reshape(1, -1).astype(int)


def getK(fX, fY, cX, cY):
	K = np.array([[fX, 0.0, cX], [0.0, fY, cY], [0.0, 0.0, 1.0]])

	return K


def getDepth(u, v, depthFile, scalingFactor):
	depth = np.asarray(Image.open(depthFile))

	return depth[v, u]/scalingFactor


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


def readDepth(depthFile):
	depth = Image.open(depthFile)
	if depth.mode != "I":
		raise Exception("Depth image is not in intensity format")

	return np.asarray(depth)


def getPointCloud(rgbFile, depthFile, K, scalingFactor):
	thresh = 10
	# thresh = 15.0

	depth = readDepth(depthFile)
	rgb = Image.open(rgbFile)

	points = []
	colors = []
	# srcPxs = []
	t0 = time.time()
	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			
			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - K[0, 2]) * Z / K[0, 0]
			Y = (v - K[1, 2]) * Z / K[1, 1]
			
			# srcPxs.append((u, v))
			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))
	t1 = time.time()
	# print("For loop",  t1-t0)

	# srcPxs = np.asarray(srcPxs).T
	points = np.asarray(points)
	colors = np.asarray(colors)
	
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	
	return pcd


def refineTrans(srcR, srcD, trgR, trgD, T2_1, K, scalingFactor):
	t0 = time.time()
	srcPcd = getPointCloud(srcR, srcD, K, scalingFactor)
	trgPcd = getPointCloud(trgR, trgD, K, scalingFactor)
	t1 = time.time()

	reg_p2l = o3d.registration.registration_colored_icp(
			srcPcd, trgPcd, 0.02, T2_1,
			o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
													relative_rmse=1e-6,
													max_iteration=30))
	t2 = time.time()
	# print("PointCloud generation: {:.2f} secs and ICP refinement: {:.2f} secs".format(t1-t0, t2-t1))

	return reg_p2l.transformation


def getIds(srcR, trgR):
	srcIdx = int(re.findall(r'\d+', os.path.basename(srcR))[0])
	trgIdx = int(re.findall(r'\d+', os.path.basename(trgR))[0])

	return srcIdx, trgIdx


def	displayCorr(u1, v1, u2, v2, srcR, trgR):
	img1 = cv2.cvtColor(np.array(Image.open(srcR)), cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(np.array(Image.open(trgR)), cv2.COLOR_BGR2RGB)

	for i in range(u1.shape[1]):
		img1 = cv2.circle(img1, (int(u1[0, i]), int(v1[0, i])), 2, (0, 0, 255), 4)
	for i in range(u1.shape[1]):
		img2 = cv2.circle(img2, (int(u2[0, i]), int(v2[0, i])), 2, (0, 0, 255), 4)

	im4 = cv2.hconcat([img1, img2])	

	# for i in range(u1.shape[1]):
	# 	im4 = cv2.line(im4, (int(u1[0, i]), int(v1[0, i])), (int(u2[0, i]) +  img1.shape[1], int(v2[0, i])), (0, 255, 0), 1)

	cv2.imshow("Image_lines", im4)
	cv2.waitKey(0)


def corr2Homo(u1, v1, u2, v2):
	uv1 = np.hstack((u1.T, v1.T))
	uv2 = np.hstack((u2.T, v2.T))

	# H, status = cv2.findHomography(uv1, uv2)
	H, status = cv2.findHomography(uv1, uv2, cv2.RANSAC, 5.0)

	# uv1 = np.vstack((uv1.T, np.ones((1, uv1.shape[0]))))
	# uv2 = np.vstack((uv2.T, np.ones((1, uv2.shape[0]))))

	# uv2Wrap = H @ uv1
	# uv2Wrap = uv2Wrap[: -1, :] / uv2Wrap[-1, :].reshape(1, -1)

	# print(uv2[: -1, 0:8])
	# print("--")
	# print(uv2Wrap[:, 0:8])

	return H


def getGtH(srcR, trgR, srcD, trgD, srcH, poses, K, scalingFactor):
	u1, v1 = getSrcPx(srcR, srcH)
	Z1 = getDepth(u1, v1, srcD, scalingFactor)
	srcIdx, trgIdx = getIds(srcR, trgR)
	T2_1 = getRelativeT(poses, srcIdx, trgIdx)
	T2_1 = refineTrans(srcR, srcD, trgR, trgD, T2_1, K, scalingFactor)

	u2, v2 = warp(u1, v1, Z1, K, T2_1)

	# displayCorr(u1, v1, u2, v2, srcR, trgR)

	gtH = corr2Homo(u1, v1, u2, v2)

	return gtH, T2_1


if __name__ == "__main__":
	np.set_printoptions(precision=3, suppress=True)

	# Realsense D455
	focalX = 382.1996765136719
	focalY = 381.8395690917969
	centerX = 312.7102355957031
	centerY = 247.72047424316406

	scalingFactor = 1000.0

	srcR = argv[1]
	srcD = argv[2]
	trgR = argv[3]
	srcH = argv[4]
	# trgH = argv[6]
	gtPoses = argv[5]
	trgD = argv[6]

	K = getK(focalX, focalY, centerX, centerY)
	poses = readPoses(gtPoses)

	gtH = getGtH(srcR, trgR, srcD, trgD, srcH, poses, K, scalingFactor)