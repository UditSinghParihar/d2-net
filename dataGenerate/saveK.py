import numpy as np


if __name__ == '__main__':
	K = np.array([[402.2965906510784, 0.0, 320.5], [0.0, 402.2965906510784, 240.5], [0.0, 0.0, 1.0]])

	np.save("K.npy", K)