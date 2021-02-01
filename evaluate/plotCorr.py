from sys import argv, exit
import csv
import matplotlib.pyplot as plt


def readData(file):
	with open(file) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')

		data = []

		for i, row in enumerate(csvReader):
			if(i == 0):
				continue
			else:
				data.append(int(row[2]))

	return data


def draw(lessThresh, moreThresh, thresh):
	labels = ['Less than {}'.format(thresh), 'More than {}'.format(thresh)]
	numCorr = [lessThresh, moreThresh]

	fig, ax = plt.subplots(figsize =(16, 9))

	plt.bar(labels, numCorr, color='maroon', width=0.4)
	plt.xlabel('Classes')
	plt.ylabel('Number of correspondences')
	plt.title('Correspondences above and below threshold.')
	plt.show()


if __name__ == '__main__':
	csvFile = argv[1]
	data = readData(csvFile)
	thresh = 15

	lessThresh = 0.0
	moreThresh = 0.0
	total = len(data)

	for e in data:
		if(e < thresh):
			lessThresh += 1
		else:
			moreThresh += 1

	print("Number of pairs with less than {} correspondences: {} ({:.2f}%). \
		\nNumber of pairs with more than {} correspondences: {} ({:.2f}%).".format(thresh, 
			lessThresh, lessThresh*100/total, thresh, moreThresh, moreThresh*100/total))

	draw(lessThresh, moreThresh, thresh)