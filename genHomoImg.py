import numpy as np
import cv2
from matplotlib import pyplot as plt
from sys import argv, exit
import os
import re


def getHomo(imgFile, outFile):
	im1 = cv2.imread(imgFile)

	# pts_floor = np.array([[180, 299], [460, 290], [585, 443], [66, 462]]) # p3dx
	pts_floor = np.array([[80, 270], [500, 270], [585, 443], [66, 462]]) # p3dx_long_homo
	pts_correct = np.array([[0, 0], [399, 0], [399, 399], [0, 399]])

	homographyMat, status = cv2.findHomography(pts_floor, pts_correct)
	img1 = cv2.warpPerspective(im1, homographyMat, (480, 300))

	# cv2.imwrite(outFile, img1)
	return im1


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


if __name__ == '__main__':
	rgbDir = argv[1]
	outDir = argv[2]

	rgbFiles = os.listdir(rgbDir)
	rgbSort = natural_sort(rgbFiles)

	# for i, rgb in enumerate(rgbSort):
	# 	img = getHomo(rgbDir + rgb, outDir + "homo{:04d}.jpg".format(i))


	imgOld = getHomo(rgbDir + rgbSort[588], outDir + "homo{:04d}.jpg".format(588))
	
	for i in range(589, len(rgbSort), 5):
		imgNew = getHomo(rgbDir + rgbSort[i], outDir + "homo{:04d}.jpg".format(i))

		sift = cv2.xfeatures2d_SURF.create(20)
		kp1, des1 = sift.detectAndCompute(imgNew,None)
		kp2, des2 = sift.detectAndCompute(imgOld,None)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1,des2,k=2)
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)

		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

		dst = cv2.warpPerspective(imgNew, H, (imgNew.shape[1]/2 + imgOld.shape[1], imgNew.shape[0]/2 + imgOld.shape[0]))
		cv2.imshow("Final0", dst)
		
		dst[0:imgOld.shape[0] , 0:imgOld.shape[1]] = imgOld
		cv2.imshow("Final", dst)
		cv2.waitKey(0)

		imgOld = dst

		# exit(1)
