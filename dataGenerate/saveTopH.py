import numpy as np
import os
import re
from sys import argv, exit
import cv2
from tqdm import tqdm


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def getWarp(imgFile, H):
	im1 = cv2.imread(imgFile)
	img1 = cv2.warpPerspective(im1, H, (800, 800))

	# cv2.imshow("Image2", img1)
	# cv2.waitKey(10)

	return img1


if __name__ == '__main__':
	rgbDir = argv[1]
	HFile = argv[2]

	rgbImgs = natural_sort(os.listdir(rgbDir))
	rgbImgs = [os.path.join(rgbDir, img) for img in rgbImgs if ".png" in img]
	H = np.load(HFile)

	for imgFile in tqdm(rgbImgs, total=len(rgbImgs)):
		warpImg = getWarp(imgFile, H)
		cv2.imwrite(imgFile, warpImg)