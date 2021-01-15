from PIL import Image, ImageOps
from sys import exit, argv
import cv2
import numpy as np


if __name__ == '__main__':
	imgFile = argv[1]
	img = Image.open(imgFile).convert('L')
	# img = ImageOps.grayscale(Image.open(imgFile))

	print(img.size, np.array(img).shape)
	
	imgNp = np.array(img)
	imgNp = imgNp[:, :, np.newaxis]
	imgNp = np.repeat(imgNp, 3, -1)


	# imgNp = cv2.cvtColor(imgNp, cv2.COLOR_BGR2RGB)
	print(imgNp.shape)
	cv2.imshow('Image', imgNp)
	cv2.waitKey(0)