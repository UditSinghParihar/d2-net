import open3d as o3d
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

		# Xtemp = X; Ytemp = Y; Ztemp = Z
		# X = Ztemp; Y = -Xtemp; Z = -Ytemp

		# print(X, Y, Z)
		trgPts.append((Z, X))

	trgPts = np.array(trgPts)
	srcPts = np.array(pts)

	# print(trgPts)
	# plotPts(trgPts)

	minX = np.min(trgPts[:, 0])
	minY = np.min(trgPts[:, 1])

	if(minX < 0):
		trgPts[:, 0] += (np.abs(minX) + 2)
	elif(minY < 0):
		trgPts[:, 1] += (np.abs(minY) + 2)
	
	maxX = np.max(trgPts[:, 0])
	maxY = np.max(trgPts[:, 1])

	# print(trgPts)
	# plotPts(trgPts)

	trgSize = 400
	ratioX = trgSize/maxX
	ratioY = trgSize/maxY

	trgPts[:, 0] *= ratioX
	trgPts[:, 1] *= ratioY

	# trgPts = trgPts.T
	# trgPts[[1, 0]] = trgPts[[0, 1]]
	# trgPts = trgPts.T
	# trgPts[[[0,1,2,3]]] = trgPts[[[3,2,1,0]]]

	print(trgPts)
	# plotPts(trgPts)
	
	for i in range(0, trgPts.shape[0]):
		rgb = cv2.circle(rgb, (int(trgPts[i, 0]), int(trgPts[i, 1])), 1, (50, 255, 50), 2)
	cv2.imshow("Image", rgb)
	cv2.waitKey(0)

	homographyMat, status = cv2.findHomography(srcPts, trgPts)

	orgImg = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
	warpImg = cv2.warpPerspective(orgImg, homographyMat, (trgSize, trgSize))

	# cv2.imshow("Warped", warpImg)
	# cv2.waitKey(0)