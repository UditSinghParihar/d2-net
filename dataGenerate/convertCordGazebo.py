# Usage: python convertCordGazebo.py poses.txt img_pairs.csv
# Input: poses.txt: Contains robot poses extracted from odometry in the form: X, Y, Theta
# Input: img_pairs.csv: Contains opposite img pairs extracted.
# Output: A matrix containing transformation of image pairs of 2nd wrt 1st camera. 
# We can use this matrix for camera 2 pose while camera 1 pose can be taken as identity. 


from sys import argv, exit
from PIL import Image
import numpy as np
import open3d as o3d
import csv
import math
import copy
import matplotlib.pyplot as plt
import os


def readPose(filename):
	f = open(filename, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []

	for i, line in enumerate(A):
		if(i % 1 == 0):
			(x, y, theta) = line.split(' ')
			# print(x, y, theta.rstrip('\n'))
			X.append(float(x))
			Y.append(float(y))
			THETA.append(math.radians(float(theta.rstrip('\n'))))

	return X, Y, THETA


def convert(x, y, theta):
	T = np.identity(4)
	T[0, 3] = x
	T[1, 3] = y
	R = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
	T[0:3, 0:3] = R
		
	return T


def draw(X, Y, THETA):
	ax = plt.subplot(111)
	ax.plot(X, Y, 'ro')
	ax.plot(X, Y, 'k-')

	ax.set_aspect('equal', 'datalim')
	ax.margins(0.1)

	plt.show()


def cameraWrtBase():
	thetaY = 0.524
	Ry = np.array([[math.cos(thetaY), 0, math.sin(thetaY)], [0, 1, 0], [-math.sin(thetaY), 0, math.cos(thetaY)]])
	To_c = np.identity(4) 
	To_c[0, 3] = 0.064
	To_c[1, 3] = -0.065
	To_c[2, 3] = 1.104
	To_c[0:3, 0:3] = Ry

	return To_c


def getPairs(filename, X, Y, THETA):
	data = []

	with open(filename) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')

		for i, row in enumerate(csvReader):
			if(i == 0):
				continue
			else:
				file1 = os.path.basename(row[0])
				file2 = os.path.basename(row[2])
				idx1 = int(''.join(filter(lambda i: i.isdigit(), file1)))
				idx2 = int(''.join(filter(lambda i: i.isdigit(), file2)))
				data.append([row[0], row[1], X[idx1], Y[idx1], THETA[idx1], 
									row[2], row[3], X[idx2], Y[idx2], THETA[idx2]])
	
	return data


def getPointCloud(rgbFile, depthFile, T=np.identity(4)):
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
	points = np.asarray(downpcd.points)
	
	# ones = np.ones((points.shape[0], 1))
	# points = np.hstack((points, ones))


	# points = T @ points.T

	# downpcd.points = o3d.utility.Vector3dVector((points.T)[:, 0:3])

	# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
	# o3d.visualization.draw_geometries([pcd, axis])
	
	return downpcd


def draw_registration_result(source, target, transformation):
	geometries = []

	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.paint_uniform_color([1, 0.706, 0])
	target_temp.paint_uniform_color([0, 0.651, 0.929])

	# X_target = Ttarget_source @ X_source
	# Ttarget_source : source wrt target
	source_temp.transform(transformation)
	geometries.append(source_temp); geometries.append(target_temp)
	
	axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2.transform(transformation)
	geometries.append(axis1); geometries.append(axis2)
	
	o3d.visualization.draw_geometries(geometries)


def right2left():
	# Right wrt left = TL_R = TR2L

	thetaX = math.radians(90)
	thetaY = math.radians(-90)

	Rx = np.array([[1, 0, 0], [0, math.cos(thetaX), -math.sin(thetaX)], [0, math.sin(thetaX), math.cos(thetaX)]])
	Ry = np.array([[math.cos(thetaY), 0, math.sin(thetaY)], [0, 1, 0], [-math.sin(thetaY), 0, math.cos(thetaY)]])

	TR2L = np.identity(4)
	TR2L[0:3, 0:3] =  Ry @ Rx 
	
	return TR2L


def visualize(data, To_c):
	for i in range(0, len(data), 50):
		rgb1, depth1, x1, y1, theta1, rgb2, depth2, x2, y2, theta2 = data[i]
		x1, y1, theta1, x2, y2, theta2 = float(x1), float(y1), float(theta1), float(x2), float(y2), float(theta2)
		
		T1 = convert(x1, y1, theta1)
		T2 = convert(x2, y2, theta2)

		pcd1 = getPointCloud(rgb1, depth1, T1)
		pcd2 = getPointCloud(rgb2, depth2, T2)
		print(i)

		# 2nd camera wrt 1st camera
		T12R = np.linalg.inv(T1 @ To_c) @ (T2 @  To_c)
		TR2L = right2left()
		T12L = TR2L @ T12R @ np.linalg.inv(TR2L)

		draw_registration_result(source=pcd2, target=pcd1, transformation=T12L)


def writeData(data, To_c):
	poses2 = np.ones((len(data), 4, 4))

	for i in range(0, len(data)):
		rgb1, depth1, x1, y1, theta1, rgb2, depth2, x2, y2, theta2 = data[i]
		x1, y1, theta1, x2, y2, theta2 = float(x1), float(y1), float(theta1), float(x2), float(y2), float(theta2)
		
		T1 = convert(x1, y1, theta1)
		T2 = convert(x2, y2, theta2)

		T12R = np.linalg.inv(T1 @ To_c) @ (T2 @  To_c)
		TR2L = right2left()
		T12L = TR2L @ T12R @ np.linalg.inv(TR2L)

		poses2[i, :, :] = np.linalg.inv(T12L)


	# World wrt 2nd camera
	rootDir = os.path.dirname(os.path.dirname(data[0][0]))
	np.save(os.path.join(rootDir, "poses2W.npy"), poses2)


if __name__ == '__main__':
	poseFile = argv[1]
	X, Y, THETA = readPose(poseFile)
	# draw(X, Y, THETA)

	pairFile = argv[2]
	data = getPairs(pairFile, X, Y, THETA)
	
	focalLength = 402.29
	centerX = 320.5
	centerY = 240.5
	scalingFactor = 1000.0

	To_c = cameraWrtBase()

	# visualize(data, To_c)

	writeData(data, To_c)