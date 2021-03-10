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
	overlapStart = 7
	overlapEnd = 14
	dbStart = 2
	dbEnd = 52

	for i in range(250, len(XWorld)-400, 3):
	# for i in range(770, len(XWorld)-1920, 1):
		row = []
		isOverStart = True
		isOverEnd = True
		isDbStart = True

		row.append(int(data[i][0]))
		for j in range(i, len(XWorld)):
			if(getDist(XWorld[i], YWorld[i], XWorld[j], YWorld[j]) > overlapStart and (isOverStart == True)):
				row.append(int(data[j][0]))
				isOverStart = False
			
			if(getDist(XWorld[i], YWorld[i], XWorld[j], YWorld[j]) > overlapEnd and (isOverEnd == True)):
				row.append(int(data[j][0]))
				isOverEnd = False

			if(getDist(XWorld[i], YWorld[i], XWorld[j], YWorld[j]) > dbStart and (isDbStart == True)):
				row.append(int(data[j][0]))
				isDbStart = False

			if(getDist(XWorld[i], YWorld[i], XWorld[j], YWorld[j]) > dbEnd):
				row.append(int(data[j][0]))
				break

		pairs.append(row)

	return pairs


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def getClosest(imgs, time):
	dist = [np.abs(time-img) for img in imgs]

	gtIdx = dist.index(min(dist))

	return imgs[gtIdx]


def getProbPairs(rearImgs, timePairs):
	rearImgs = [int(img.replace('.png', '')) for img in rearImgs]
	
	frontTime = [time[0] for time in timePairs]
	rearGtSt = [time[2] for time in timePairs]
	rearGtEnd = [time[3] for time in timePairs]
	rearDbSt = [time[1] for time in timePairs]
	rearDbEnd = [time[4] for time in timePairs]

	imgPairs = []

	for i in range(len(frontTime)):
		frontImg = os.path.join(frontDir, str(frontTime[i]) + ".png")

		gtImgSt = getClosest(rearImgs, rearGtSt[i])
		gtImgEnd = getClosest(rearImgs, rearGtEnd[i])
		dbImgSt = getClosest(rearImgs, rearDbSt[i])
		dbImgEnd = getClosest(rearImgs, rearDbEnd[i])

		gtImgSt = os.path.join(rearDir, str(gtImgSt) + ".png")
		gtImgEnd = os.path.join(rearDir, str(gtImgEnd) + ".png")
		dbImgSt = os.path.join(rearDir, str(dbImgSt) + ".png")
		dbImgEnd = os.path.join(rearDir, str(dbImgEnd) + ".png")

		imgPairs.append((frontImg, gtImgSt, gtImgEnd, dbImgSt, dbImgEnd))

	return imgPairs


def writePairs(pairs):
	with open('gtPairsHSub.csv', 'w', newline='') as file:
		writer = csv.writer(file)

		# title = ['front', 'rearStart', 'rearEnd']
		title = ['frontQuery', 'gtRearStart', 'gtRearEnd', 'dbRearStart', 'dbRearEnd']

		writer.writerow(title)

		for pair in pairs:
			writer.writerow(list(pair))


if __name__ == '__main__':
	csvFile = argv[1]
	frontDir = argv[2]
	rearDir = argv[3]
	
	data = readData(csvFile)
	trans = convert2World(data)

	XWorld, YWorld, ZWorld = getXYZ(trans)
	# draw(XWorld, YWorld)

	timePairs = getTimePairs(XWorld, YWorld, data)

	frontImgs = natural_sort([file for file in os.listdir(frontDir) if '.png' in file])
	rearImgs = natural_sort([file for file in os.listdir(rearDir) if '.png' in file])

	imgPairs = getProbPairs(rearImgs, timePairs)

	writePairs(imgPairs)