from sys import argv, exit
import cv2 
from matplotlib import pyplot as plt
import os
import re


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def plotComb(d2netImgs, rordImgs, d2netPerImgs, d2netErr, rordErr, d2netPerErr):
	print(len(d2netErr))
	fig = plt.figure(figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')
	rows, cols = 3, 1

	for i in range(len(rordImgs)):
		fig.add_subplot(rows, cols, 1)
		plt.imshow(d2netPerImgs[i])
		plt.axis('off')
		plt.title('D2-Net', fontsize=12, rotation='vertical',x=-0.05, y=0.6)
		plt.text(x=1300, y=250, s=d2netPerErr[i], size=12, verticalalignment='center')


		fig.add_subplot(rows, cols, 2)
		plt.imshow(d2netImgs[i])
		plt.axis('off')
		plt.title('D2-Net + OV', fontsize=12, rotation='vertical',x=-0.05, y=0.78)
		plt.text(x=1300, y=250, s=d2netErr[i], size=12, verticalalignment='center')

		fig.add_subplot(rows, cols, 3)
		plt.imshow(rordImgs[i])
		plt.axis('off')
		plt.title('RoRD + OV\n(Ours)', fontsize=12, rotation='vertical',x=-0.05, y=0.7)
		plt.text(x=1300, y=250, s=rordErr[i], size=12, verticalalignment='center')

		plt.subplots_adjust(wspace=0.01, hspace=0.01)
		plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.01)

		plt.show(block=False)
		plt.pause(0.001)
		plt.clf()


def readError(txt):
	f = open(txt, 'r')
	A = f.readlines()
	f.close()

	errors = []
	for line in A:
		idx, error = line.split(' ')
		st = "Rotation Error: {:.2f}".format(float(error.rstrip('\n')))
		errors.append(st)

	return errors


if __name__ == '__main__':
	d2netDir = argv[1] 
	rordDir = argv[2]
	d2netPerDir = argv[3]

	d2netFiles = natural_sort([os.path.join(d2netDir, file) for file in os.listdir(d2netDir) if '.jpg' in file])
	rordFiles = natural_sort([os.path.join(rordDir, file) for file in os.listdir(rordDir) if '.jpg' in file])
	d2netPerFiles = natural_sort([os.path.join(d2netPerDir, file) for file in os.listdir(rordDir) if '.jpg' in file])

	d2netTxt = os.path.join(d2netDir, "R_norm_A.txt")
	rordTxt = os.path.join(rordDir, "R_norm_A.txt")
	d2netPerTxt = os.path.join(d2netPerDir, "R_norm_A.txt")

	d2netErr = readError(d2netTxt)
	rordErr = readError(rordTxt)
	d2netPerErr = readError(d2netPerTxt)

	d2netImgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in d2netFiles]
	rordImgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in rordFiles]
	d2netPerImgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in d2netPerFiles]
	print("Images loaded.")

	plotComb(d2netImgs, rordImgs, d2netPerImgs, d2netErr, rordErr, d2netPerErr)

	# for i in range(len(rordImgs)):
	# 	cv2.imshow("D2Net", d2netImgs[i])
	# 	cv2.imshow("RoRD", rordImgs[i])
	# 	cv2.waitKey(100)