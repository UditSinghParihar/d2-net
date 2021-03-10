from sys import argv, exit
import csv
import matplotlib.pyplot as plt
import os
from extractMatchTopRobo import getPerspKeypoints
import cv2


def readCSV(file):
	with open(file) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')

		data = []

		for i, row in enumerate(csvReader):
			if(i == 0):
				continue
			else:
				data.append(row)

	return data


if __name__ == '__main__':
	vprFile = argv[1]
	gtFile = argv[2]

	vprData = readCSV(vprFile)
	gtData = readCSV(gtFile)

	# WEIGHTS = "/home/udit/udit/d2-net/results/train_corr20_robotcar_H_same/checkpoints/d2.15.pth"
	WEIGHTS = "models/d2_tf.pth"

	# outDir = "/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/subSeq/d2net/"
	outDir = "/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/subSeq/sift/"
	# outDir = "/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/subSeq/rord/"

	srcH = "dataGenerate/frontHomo4.npy"
	trgH = "dataGenerate/rearHomo.npy"

	frontDir = "/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/stereo/centre_rgb/"
	rearDir = "/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/mono_rear_rgb/"

	font = cv2.FONT_HERSHEY_SIMPLEX
	org = (600, 50)
	orgQuery = (10, 50)
	colorQuery = (255, 0, 0)
	fontScale = 2
	thickness = 5
	queryStr = "Query"

	correctPair = 0
	imgIdx = 0
	for vpr, gt in zip(vprData, gtData):
		predId = int(vpr[1])
		
		gtStId = int(os.path.basename(gt[1]).replace('.png', ''))
		gtEndId = int(os.path.basename(gt[2]).replace('.png', ''))
		
		inlierStr = "Wrong Retrieval"
		color = (0, 0, 255)

		if(gtStId < predId < gtEndId):
			correctPair += 1
			inlierStr = "Right Retrieval"
			color = (0, 255, 0)

		srcR = os.path.join(frontDir, os.path.basename(vpr[0]))
		trgR = os.path.join(rearDir, vpr[1] + ".png")
		# print(srcR, trgR)
		topMatchImg, perpMatchImg = getPerspKeypoints(srcR, trgR, srcH, trgH, WEIGHTS)

		perpMatchImg = cv2.putText(perpMatchImg, inlierStr, org, font,  
							fontScale, color, thickness, cv2.LINE_AA)

		perpMatchImg = cv2.putText(perpMatchImg, queryStr, orgQuery, font,  
							fontScale, colorQuery, thickness, cv2.LINE_AA) 

		# cv2.imshow("Image", perpMatchImg)
		# cv2.waitKey(10)

		outFile = os.path.join(outDir, str(imgIdx)+'.png')
		cv2.imwrite(outFile, perpMatchImg)

		imgIdx += 1

	print("Number of correct retrived top-1 pair: {} out of total {} pairs ({:.2f}%)".format(correctPair, 
			len(vprData), correctPair*100.0/len(vprData)))