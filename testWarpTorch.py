import numpy as np
import cv2
from matplotlib import pyplot as plt
from sys import argv, exit


if __name__ == '__main__':
	imgFile1 = "/home/cair/backup/d2-net/data_gazebo/data5/rgb/rgb000318.jpg"
	imgFile2 = "/home/cair/backup/d2-net/data_gazebo/data5/rgb/rgb001439.jpg"

	orgSrc = np.load("pos1_before.npy")
	orgDst = np.load("pos2_before.npy")

	im1 = cv2.imread(imgFile1)
	im2 = cv2.imread(imgFile2)

	for i in range(0, orgSrc.shape[1], 50):
		im1 = cv2.circle(im1, (int(orgSrc[1, i]), int(orgSrc[0, i])), 1, (0, 0, 255), 2)
	for i in range(0, orgDst.shape[1], 50):
		im2 = cv2.circle(im2, (int(orgDst[1, i]), int(orgDst[0, i])), 1, (0, 0, 255), 2)

	im4 = cv2.hconcat([im1, im2])
	
	for i in range(0, orgSrc.shape[1], 50):
		im4 = cv2.line(im4, (int(orgSrc[1, i]), int(orgSrc[0, i])), (int(orgDst[1, i]) +  im1.shape[1], int(orgDst[0, i])), (0, 255, 0), 1)
	
	# cv2.imshow("Image_lines", im4)
	# cv2.waitKey(0)


	pts_floor = np.array([[190,210],[455,210],[633,475],[0,475]])
	pts_correct = np.array([[0, 0], [399, 0], [399, 399], [0, 399]])
	homographyMat, status = cv2.findHomography(pts_floor, pts_correct)
	
	startPts = np.array([[0, 0], [400, 0], [400, 400], [0, 400]])
	endPts = np.array([[400, 400], [0, 400], [0, 0], [400, 0]])
	homoFl, status = cv2.findHomography(startPts, endPts)

	img1 = cv2.warpPerspective(cv2.imread(imgFile1), homographyMat, (400, 400))
	img2 = cv2.warpPerspective(cv2.imread(imgFile2), np.dot(homoFl, homographyMat), (400, 400))

	ones = np.ones((1, orgSrc.shape[1]))
	srcHomo = np.vstack((orgSrc, ones))
	dstHomo = np.vstack((orgDst, ones))

	srcWarp = np.dot(homographyMat, srcHomo)
	dstWarp = np.dot(np.dot(homoFl, homographyMat), dstHomo)

	srcWarp = srcWarp/srcWarp[2, :]
	dstWarp = dstWarp/dstWarp[2, :]

	warpSrc = srcWarp[0:2, :]
	warpDst = dstWarp[0:2, :]

	srcPov = []
	dstPov = []
	for i in range(warpSrc.shape[1]):
		if(400 > warpSrc[0, i] > 0 and 400 > warpSrc[1, i] > 0 and 400 > warpDst[0, i] > 0 and 400 > warpDst[1, i] > 0):
			srcPov.append((warpSrc[0, i], warpSrc[1, i]))
			dstPov.append((warpDst[0, i], warpDst[1, i]))

	srcPov = np.asarray(srcPov).T
	dstPov = np.asarray(dstPov).T
	
	for i in range(0, srcPov.shape[1], 100):
		img1 = cv2.circle(img1, (int(srcPov[1, i]), int(srcPov[0, i])), 1, (0, 0, 255), 2)
	for i in range(0, dstPov.shape[1], 100):
		img2 = cv2.circle(img2, (int(dstPov[1, i]), int(dstPov[0, i])), 1, (0, 0, 255), 2)

	im4 = cv2.hconcat([img1, img2])

	for i in range(0, srcPov.shape[1], 100):
		im4 = cv2.line(im4, (int(srcPov[1, i]), int(srcPov[0, i])), (int(dstPov[1, i]) +  img1.shape[1], int(dstPov[0, i])), (0, 255, 0), 1)

	cv2.imshow("Image_lines", im4)
	cv2.waitKey(0)