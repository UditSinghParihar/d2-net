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
	dist = 7.8

	for i in range(200, len(XWorld)-80):
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
	# print(len(imgs), time)
	dist = [np.abs(time-img) for img in imgs]

	return imgs[dist.index(min(dist))]


def getImgPairs(frontImgs, rearImgs, timePairs):
	frontImgs = [int(img.replace('.png', '')) for img in frontImgs]
	rearImgs = [int(img.replace('.png', '')) for img in rearImgs]
	
	frontTime = [time[0] for time in timePairs]
	rearTime = [time[1] for time in timePairs]

	imgPairs = []

	for i in range(len(frontTime)):
		frontImg = str(frontTime[i]) + ".png"
		rearImg = str(getClosest(rearImgs, rearTime[i])) + ".png"

		imgPairs.append((os.path.join(frontDir, frontImg), os.path.join(rearDir, rearImg)))

	return imgPairs


def writePairs(pairs):
	with open('imagePairsOxford.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['front', 'rear'])

		for pair in pairs:
			writer.writerow([pair[0], pair[1]])


def drawPairs(pairs):
	fig, ax = plt.subplots(1, 2)

	for pair in pairs:
		imf = Image.open(pair[0])
		imr = Image.open(pair[1])

		ax[0].imshow(imf)
		ax[1].imshow(imr)
		plt.pause(0.000001)


if __name__ == '__main__':
	csvFile = argv[1]
	frontDir = argv[2]
	rearDir = argv[3]
	
	data = readData(csvFile)
	trans = convert2World(data)

	XWorld, YWorld, ZWorld = getXYZ(trans)
	draw(XWorld, YWorld)

	timePairs = getTimePairs(XWorld, YWorld, data)

	frontImgs = natural_sort(os.listdir(frontDir))
	rearImgs = natural_sort(os.listdir(rearDir))

	imgPairs = getImgPairs(frontImgs, rearImgs, timePairs)

	writePairs(imgPairs)

	drawPairs(imgPairs)	
