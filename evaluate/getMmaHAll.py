from sys import argv, exit
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from getMmaH import getGtH


def readPoses(file):
	f = open(file, 'r')
	A = f.readlines()
	f.close()

	poses = []

	for i, line in enumerate(A):
		T = np.identity(4)
		row = line.split(' ')
		px, py, pz, qx, qy, qz, qw = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
		if (px==py==pz==qx==qy==qz==qw == 0):
			poses.append(np.identity(4))
			continue
		Rot = R.from_quat([qx, qy, qz, qw])
		T[0:3, 0:3] = Rot.as_dcm()
		T[0, 3] = px
		T[1, 3] = py
		T[2, 3] = pz

		poses.append(T)

	return poses


def getK(fX, fY, cX, cY):
	K = np.array([[fX, 0.0, cX], [0.0, fY, cY], [0.0, 0.0, 1.0]])

	return K


if __name__ == '__main__':
	np.set_printoptions(precision=3, suppress=True)

	gtPoses = argv[1]
	rgb_csv = argv[2]
	depth_csv = argv[3]

	# Realsense D455
	focalX = 382.1996765136719
	focalY = 381.8395690917969
	centerX = 312.7102355957031
	centerY = 247.72047424316406
	scalingFactor = 1000.0

	K = getK(focalX, focalY, centerX, centerY)
	poses = readPoses(gtPoses)

	df_rgb = pd.read_csv(rgb_csv)
	df_dep = pd.read_csv(depth_csv)

	i = 0
	for im_q, dep_q in zip(df_rgb['query'], df_dep['query']):
		filter_list_H = []
		filter_list_T = []
		H_q = im_q.replace('jpg', 'npy')
		
		for im_d, dep_d in zip(df_rgb.iteritems(), df_dep.iteritems()):
			if im_d[0] == 'query':
				continue

			gtH, gtT = getGtH(im_q, im_d[1][1], dep_q, dep_d[1][1], H_q, poses, K, scalingFactor)

			filter_list_H.append(gtH)
			filter_list_T.append(gtT)
			print("Extracted database.")

		print("Extracted query: {}".format(i))

		HQuery = np.stack(filter_list_H).transpose(1,2,0)
		np.save('/scratch/udit/realsense/dataVO/data5/HRefine/' + str(i) + '.npy', HQuery)

		TQuery = np.stack(filter_list_T).transpose(1,2,0)
		np.save('/scratch/udit/realsense/dataVO/data5/TRefine/' + str(i) + '.npy', TQuery)
		i += 1
