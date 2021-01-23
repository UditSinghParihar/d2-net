from sys import argv, exit
import math
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import re
from PIL import Image


def readData(file):
	with open(file) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')

		data = []

		for i, row in enumerate(csvReader):
			if(i == 0):
				continue
			else:
				data.append(row)

	return data


def getTrans(row):
	srcTime, dstTime, x, y, z, roll, pitch, yaw = row
	srcTime, dstTime, x, y, z, roll, pitch, yaw = float(srcTime), float(dstTime), float(x), float(y), float(z), float(roll), float(pitch), float(yaw)
	r = R.from_euler('zyx', [yaw, pitch, roll])

	T = np.identity(4)
	T[0, 3] = x
	T[1, 3] = y
	T[2, 3] = z
	T[0:3, 0:3] = r.as_dcm()

	return np.linalg.inv(T)
	# return T


def getXYZ(trans):
	X, Y, Z = [], [], []

	for T in trans:
		x, y, z = T[0, 3], T[1, 3], T[2, 3]
		X.append(x); Y.append(y); Z.append(z)

	return X, Y, Z


def convert2World(data):
	posesWorld = []

	curWorld = getTrans(data[0])

	posesWorld.append(curWorld)
	prevWorld = curWorld

	for i in range(1, len(data)):
		curLocal = getTrans(data[i])
		curWorld = prevWorld @ curLocal

		posesWorld.append(curWorld)
		prevWorld = curWorld

	return posesWorld


def draw(X, Y):
	ax = plt.subplot(111)
	ax.plot(X, Y, 'ro')
	ax.plot(X, Y, 'k-')

	plt.show()


def getDist(x1, y1, x2, y2):
	return ((x1-x2)**2 + (y1-y2)**2)**0.5


def getTimePairs(XWorld, YWorld, data):
	pairs = []
	dist = 9

	for i in range(250, len(XWorld)-250, 5):
		for j in range(i, len(XWorld)):
			if(getDist(XWorld[i], YWorld[i], XWorld[j], YWorld[j]) > dist):
				pairs.append((int(data[i][0]), int(data[j][0])))
				break

	return pairs


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def getClosest(imgs, time):
	dist = [np.abs(time-img) for img in imgs]

	probMatches = []

	gtIdx = dist.index(min(dist))
	probMatches.append(imgs[gtIdx])

	for idx in range(gtIdx, gtIdx-125, -5):
		if(idx<0 or idx>=len(imgs)):
			continue
		
		probMatches.append(imgs[idx])

	for idx in range(gtIdx, gtIdx+125, 5):
		if(idx<0 or idx>=len(imgs)):
			continue

		probMatches.append(imgs[idx])

	return imgs[gtIdx], probMatches


def getProbPairs(frontImgs, rearImgs, timePairs):
	frontImgs = [int(img.replace('.png', '')) for img in frontImgs]
	rearImgs = [int(img.replace('.png', '')) for img in rearImgs]
	
	frontTime = [time[0] for time in timePairs]
	rearTime = [time[1] for time in timePairs]

	imgPairs = []

	for i in range(len(frontTime)):
		frontImg = str(frontTime[i]) + ".png"

		gtImg, probMatches = getClosest(rearImgs, rearTime[i])

		probImgs = []
		for time in probMatches:
			rearImg = str(time) + ".png"
			probImgs.append(os.path.join(rearDir, rearImg))

		imgPairs.append((os.path.join(frontDir, frontImg), probImgs))

	return imgPairs


def writePairs(pairs):
	with open('probPairs.csv', 'w', newline='') as file:
		writer = csv.writer(file)

		title = []
		title.append('front')

		numRear = len(pairs[0][1])
		for i in range(numRear):
			title.append('rear' + str(i))

		writer.writerow(title)

		for pair in pairs:
			row = []
			row.append(pair[0])

			for img in pair[1]:
				row.append(img)

			writer.writerow(row)


if __name__ == '__main__':
	csvFile = argv[1]
	frontDir = argv[2]
	rearDir = argv[3]
	
	data = readData(csvFile)
	trans = convert2World(data)

	XWorld, YWorld, ZWorld = getXYZ(trans)
	# draw(XWorld, YWorld)

	timePairs = getTimePairs(XWorld, YWorld, data)

	frontImgs = natural_sort(os.listdir(frontDir))
	rearImgs = natural_sort(os.listdir(rearDir))

	imgPairs = getProbPairs(frontImgs, rearImgs, timePairs)

	writePairs(imgPairs)

	# drawPairs(imgPairs)	
