from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv, exit
import os


ImgPairs = pd.read_csv(argv[1], usecols = ['front','rear'])
fig, ax = plt.subplots(1, 2)

frontDir = '/scratch/udit/robotcar/overcast/ipm3/front_top/'
rearDir = '/scratch/udit/robotcar/overcast/ipm3/rear_top/'

for idx, row in ImgPairs.iterrows():
		# imf = Image.open(row['front'])
		# imr = Image.open(row['rear'])

		# ax[0].imshow(imf)
		# ax[1].imshow(imr)
		# plt.pause(0.00001)
	
		if(idx%15 == 0):
			frontFile = row['front'] 
			rearFile = row['rear']

			cmd1 = 'cp' + ' ' + frontFile + ' ' + frontDir + str(idx//15) + '.png'
			cmd2 = 'cp' + ' ' + rearFile + ' ' + rearDir + str(idx//15) + '.png' 

			os.system(cmd1)
			os.system(cmd2)
