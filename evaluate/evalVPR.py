from sys import argv, exit
import csv
import matplotlib.pyplot as plt


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
		if(vpr[1] == gt[1]):
			correctPair += 1

	print("Number of correct retrived top-1 pair: {} out of total {} pairs ({:.2f}%)".format(correctPair, 
			len(vprData), correctPair*100.0/len(vprData)))