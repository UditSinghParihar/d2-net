from sys import argv, exit
import csv
import matplotlib.pyplot as plt
import os


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
	gtFile = argv[1]
	vprFile = argv[2]


	gtData = readCSV(gtFile)
	vprData = readCSV(vprFile)

	correctPair = 0

	for vpr, gt in zip(vprData, gtData):
		predId = int(vpr[1])
		
		gtStId = int(os.path.basename(gt[1]).replace('.png', ''))
		gtEndId = int(os.path.basename(gt[2]).replace('.png', ''))

		if(gtStId < predId < gtEndId):
			correctPair += 1

	print("Number of correct retrived top-1 pair: {} out of total {} pairs ({:.2f}%)".format(correctPair, 
			len(vprData), correctPair*100.0/len(vprData)))


