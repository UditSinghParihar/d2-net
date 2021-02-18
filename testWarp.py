import numpy as np
import cv2
from matplotlib import pyplot as plt
from sys import argv, exit


def getCorr(imgFile1, imgFile2):
	MIN_MATCH_COUNT = 1

	im1 = cv2.imread(imgFile1)
	im2 = cv2.imread(imgFile2)

	# Front1
	# pts_correct = np.array([[300, 800], [600, 800], [600, 0], [300, 0]])
	# pts_floor = np.array([[139,802], [1275,792], [799,508], [501,519]])

	# Front2
	# pts_correct = np.array([[100, 700], [700, 700], [700, 0], [100, 0]])
	# pts_floor = np.array([[139,802], [1275,792], [922,578], [425,578]])

	# # Front3
	# pts_correct = np.array([[100, 770], [700, 770], [700, 170], [100, 170]])
	# pts_floor = np.array([[139,802], [1275,792], [1007,619], [373,618]])

	# # Front4
	# pts_correct = np.array([[50, 630], [750, 630], [750, 170], [50, 170]])
	pts_correct = np.array([[50, 770], [750, 770], [750, 310], [50, 310]])
	pts_floor = np.array([[139,802], [1275,792], [1099,675], [337,649]])

	# Rear1
	# pts_correct = np.array([[300, 800], [750, 800], [750, 0], [300, 0]])
	# pts_floor = np.array([[70,657], [1013,658], [668,496], [366,487]])

	# # Rear2
	# pts_correct = np.array([[200, 670], [800, 670], [800, 70], [200, 70]])
	# pts_floor = np.array([[3,792], [1011,789], [686,547], [334,548]])

	homographyMat, status = cv2.findHomography(pts_floor, pts_correct)
	img1 = cv2.warpPerspective(im1, homographyMat, (800, 800))
	# np.save("dataGenerate/frontHomo.npy", homographyMat)
	
	# pts_floor2 = np.array([[190,210],[455,210],[633,475],[0,475]])
	# homographyMat, status = cv2.findHomography(pts_floor2, pts_correct)
	# img2 = cv2.warpPerspective(im2, homographyMat, (800, 800))


	cv2.namedWindow('Image2', cv2.WINDOW_NORMAL)

	cv2.imshow("Image1", img1)
	cv2.imshow("Image2", im1)
	cv2.waitKey(0)
	exit(1)

	startPts = np.array([[0, 0], [400, 0], [400, 400], [0, 400]])
	endPts = np.array([[400, 400], [0, 400], [0, 0], [400, 0]])
	homoFl, status = cv2.findHomography(startPts, endPts)
	imgFl2 = cv2.warpPerspective(img2, homoFl, (400, 400))

	# cv2.imshow("Image1", img1)
	# cv2.imshow("Image2", imgFl2)
	# cv2.waitKey(0)

	img2 = imgFl2
	# exit(1)
	# img1 = im1
	# img2 = im2

	sift = cv2.xfeatures2d_SURF.create(20)
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
	good = good[0:16]
	print("Number of matches: {}".format(len(good)))

	print(type(good), len(good), type(good[0]))

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		draw_params = dict(matchColor = (0,255,0),
						   singlePointColor = None,
						   matchesMask = matchesMask,
						   flags = 2)
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
		# cv2.imshow('Matches', img3)
		# cv2.waitKey(0)

	else:
		print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		draw_params = dict(matchColor = (0,255,0),
						   singlePointColor = None,
						   matchesMask = matchesMask,
						   flags = 2)
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
		cv2.imshow('Matches', img3)
		cv2.waitKey(0)

		exit(1)

	src_pts = src_pts.squeeze(); dst_pts = dst_pts.squeeze()
	ones = np.ones((src_pts.shape[0], 1))

	src_pts = np.hstack((src_pts, ones))
	dst_pts = np.hstack((dst_pts, ones))

	orgSrc = np.dot(np.linalg.inv(homographyMat), src_pts.T)
	orgDst = np.dot(np.dot(np.linalg.inv(homographyMat), np.linalg.inv(homoFl)), dst_pts.T)

	orgSrc = orgSrc/orgSrc[2, :]
	orgDst = orgDst/orgDst[2, :]

	print("Showing Original Image Pairs")

	imRgb1 = cv2.imread(imgFile1)
	imRgb2 = cv2.imread(imgFile2)

	for i in range(orgSrc.shape[1]):
		im1 = cv2.circle(imRgb1, (int(orgSrc[0, i]), int(orgSrc[1, i])), 5, (0, 0, 255), 3)
	for i in range(orgDst.shape[1]):
		im2 = cv2.circle(imRgb2, (int(orgDst[0, i]), int(orgDst[1, i])), 5, (0, 0, 255), 3)
	
	im4 = cv2.hconcat([im1, im2])
	
	for i in range(orgSrc.shape[1]):
		im4 = cv2.line(im4, (int(orgSrc[0, i]), int(orgSrc[1, i])), (int(orgDst[0, i]) +  im1.shape[1], int(orgDst[1, i])), (0, 255, 0), 1)
	
	cv2.imshow("Image_lines", im4)
	cv2.waitKey(0)

	return orgSrc, orgDst, homographyMat, homoFl


def warp(imgFile1, imgFile2, orgSrc, orgDst, HCrop, HFlip):
	im1 = cv2.imread(imgFile1)
	im2 = cv2.imread(imgFile2)

	img1 = cv2.warpPerspective(im1, HCrop, (400, 400))
	img2 = cv2.warpPerspective(im2, HCrop, (400, 400))

	# cv2.imshow("Image1", img1)
	# cv2.imshow("Image2", img2)
	# cv2.waitKey(0)

	warpSrc = np.dot(HCrop, orgSrc)
	warpDst = np.dot(HCrop, orgDst)

	warpSrc = warpSrc/warpSrc[2, :]
	warpDst = warpDst/warpDst[2, :]

	for i in range(warpSrc.shape[1]):
		img1 = cv2.circle(img1, (int(warpSrc[0, i]), int(warpSrc[1, i])), 5, (0, 0, 255), 3)
	for i in range(warpDst.shape[1]):
		img2 = cv2.circle(img2, (int(warpDst[0, i]), int(warpDst[1, i])), 5, (0, 0, 255), 3)

	im4 = cv2.hconcat([img1, img2])

	for i in range(warpSrc.shape[1]):
		im4 = cv2.line(im4, (int(warpSrc[0, i]), int(warpSrc[1, i])), (int(warpDst[0, i]) +  img1.shape[1], int(warpDst[1, i])), (0, 255, 0), 1)

	cv2.imshow("Image_lines", im4)
	cv2.waitKey(0)


if __name__ == '__main__':
	imgFile1 = argv[1]
	imgFile2 = argv[2]

	orgSrc, orgDst, HCrop, HFlip = getCorr(imgFile1, imgFile2)

	warp(imgFile1, imgFile2, orgSrc, orgDst, HCrop, HFlip)
	
	# orgSrc = np.asarray(orgSrc)[0:2, :]
	# orgDst = np.asarray(orgDst)[0:2, :]

	# np.savetxt('src_pts.txt', orgSrc, delimiter=' ')
	# np.savetxt('trg_pts.txt', orgDst, delimiter=' ')