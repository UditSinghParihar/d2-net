from sys import argv, exit
import cv2 
from matplotlib import pyplot as plt
import os
import re


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def plotComb(d2netImgs, rordImgs, siftImgs):
	fig = plt.figure(figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')
	rows, cols = 3, 1

	for i in range(len(rordImgs)):
		fig.add_subplot(rows, cols, 1)
		plt.imshow(d2netImgs[i])
		plt.axis('off')
		plt.title('D2-Net + OV', fontsize=12, rotation='vertical',x=-0.05, y=0.7)

		fig.add_subplot(rows, cols, 2)
		plt.imshow(siftImgs[i])
		plt.axis('off')
		plt.title('SIFT + OV', fontsize=12, rotation='vertical',x=-0.05, y=0.7)

		fig.add_subplot(rows, cols, 3)
		plt.imshow(rordImgs[i])
		plt.axis('off')
		plt.title('RoRD + OV\n(Ours)', fontsize=12, rotation='vertical',x=-0.05, y=0.7)

		plt.subplots_adjust(wspace=0.01, hspace=0.01)
		# fig.tight_layout()
		plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.01)

		plt.show(block=False)
		plt.pause(0.001)
		plt.clf()


if __name__ == '__main__':
	d2netDir = argv[1] 
	rordDir = argv[2]
	siftDir = argv[3]

	d2netFiles = natural_sort([os.path.join(d2netDir, file) for file in os.listdir(d2netDir) if '.png' in file])
	rordFiles = natural_sort([os.path.join(rordDir, file) for file in os.listdir(rordDir) if '.png' in file])
	siftFiles = natural_sort([os.path.join(siftDir, file) for file in os.listdir(siftDir) if '.png' in file])

	# d2netImgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in d2netFiles]
	# rordImgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in rordFiles]
	# siftImgs = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in siftFiles]

	d2netImgs = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), (1600, 600)) for file in d2netFiles]
	rordImgs = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), (1600, 600)) for file in rordFiles]
	siftImgs = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), (1600, 600)) for file in siftFiles]

	print("Images loaded.")

	plotComb(d2netImgs, rordImgs, siftImgs)

	# for i in range(len(rordImgs)):
	# 	cv2.imshow("D2Net", d2netImgs[i])
	# 	cv2.imshow("RoRD", rordImgs[i])
	# 	cv2.waitKey(100)