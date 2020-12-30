from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv, exit


ImgPairs = pd.read_csv(argv[1], usecols = ['front','rear'])
fig, ax = plt.subplots(1, 2)

for idx, row in ImgPairs.iterrows():
		imf = Image.open(row['front'])
		imr = Image.open(row['rear'])

		ax[0].imshow(imf)
		ax[1].imshow(imr)
		plt.pause(0.00001)
