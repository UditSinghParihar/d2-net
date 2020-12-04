# import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plotPts(trgPts):
	ax = plt.subplot(111)
	ax.plot(trgPts[:, 1], trgPts[:, 0], 'ro')
	plt.show()


if __name__ == '__main__':
	rgbFile = argv[1]
	depthFile = argv[2]

	srcPts = []
	trgPts = []

	depth = np.load(depthFile)
	img = Image.open(rgbFile)

	# bottom left -> bottom right -> top right -> top left 
	pts = [[23, 406], [597, 393], [522, 145], [98, 144]]

	rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
	for i in range(0, len(pts)):
		rgb = cv2.circle(rgb, (pts[i][0], pts[i][1]), 1, (0, 0, 255), 2)
	# cv2.imshow("Image", rgb)
	# cv2.waitKey(0)

	scalingFactor = 1000.0
	focalLength = 402.29
	centerX = 320.5
	centerY = 240.5

	for u, v in pts:
		Z = depth[v, u]/scalingFactor
		X = (u - centerX) * Z / focalLength
		Y = (v - centerY) * Z / focalLength

		trgPts.append((X, Z))

	trgPts = np.array(trgPts)
	srcPts = np.array(pts)

	# 3D point adjustment to opencv plane
	# trgPts[2, 1], trgPts[3, 1] = -trgPts[2, 1], -trgPts[3, 1]
	# trgPts[:, 1] = -trgPts[:, 1]

	# Making coordinates positive
	minX = np.min(trgPts[:, 0])
	minY = np.min(trgPts[:, 1])

	if(minX < 0):
		trgPts[:, 0] += (np.abs(minX) + 2)
	if(minY < 0):
		trgPts[:, 1] += (np.abs(minY) + 2)
	
	# Scaling coordinates
	maxX = np.max(trgPts[:, 0])
	maxY = np.max(trgPts[:, 1])

	trgSize = 400
	ratioX = trgSize/maxX
	ratioY = trgSize/maxY

	trgPts[:, 0] *= ratioX
	trgPts[:, 1] *= ratioY

	# print(trgPts)
	# plotPts(trgPts)

	for i in range(0, trgPts.shape[0]):
		rgb = cv2.circle(rgb, (int(trgPts[i, 0]), int(trgPts[i, 1])), 1, (50, 255, 50), 2)
	cv2.imshow("Image", rgb)
	cv2.waitKey(0)

	homographyMat, status = cv2.findHomography(srcPts, trgPts)
	orgImg = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
	warpImg = cv2.warpPerspective(orgImg, homographyMat, (trgSize, trgSize))

	cv2.imshow("Warped", warpImg)
	cv2.waitKey(0)