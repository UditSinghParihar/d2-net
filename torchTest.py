from sys import argv, exit
import torch
import torchgeometry as tgm
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn


def rotate(imgFile):
	image = cv2.imread(imgFile)[..., (2,1,0)]
	print(image.shape)

	img = tgm.utils.image_to_tensor(image)
	img = torch.unsqueeze(img.float(), dim=0)


	alpha = 45.0  # in degrees
	angle = torch.ones(1) * alpha

	# define the rotation center
	center = torch.ones(1, 2)
	center[..., 0] = img.shape[3] / 2  # x
	center[..., 1] = img.shape[2] / 2  # y

	# define the scale factor
	scale = torch.ones(1)

	# compute the transformation matrix
	M = tgm.get_rotation_matrix2d(center, angle, scale)

	# apply the transformation to original image
	_, _, h, w = img.shape
	img_warped = tgm.warp_affine(img, M, dsize=(h, w))

	# print(img_warped.byte().shape)
	# exit(1)

	# convert back to numpy
	image_warped = tgm.utils.tensor_to_image(torch.squeeze(img_warped.byte(), dim=0))

	fig, axs = plt.subplots(1, 2, figsize=(16, 10))
	axs = axs.ravel()

	axs[0].axis('off')
	axs[0].set_title('image source')
	axs[0].imshow(image)

	axs[1].axis('off')
	axs[1].set_title('image warped')
	axs[1].imshow(image_warped)

	plt.show()


def warp(imgFile):
	image = cv2.imread(imgFile)[..., (2,1,0)]
	print(image.shape)

	img = tgm.utils.image_to_tensor(image)
	img = torch.unsqueeze(img.float(), dim=0)  # BxCxHxW

	points_src = torch.FloatTensor([[
		[190,210],[455,210],[633,475],[0,475],
	]])

	points_dst = torch.FloatTensor([[
		[0, 0], [399, 0], [399, 399], [0, 399],
	]])

	M1 = tgm.get_perspective_transform(points_src, points_dst)
	# img_warp = tgm.warp_perspective(img, M1, dsize=(400, 400))

	points_src = torch.FloatTensor([[
		[0, 0], [400, 0], [400, 400], [0, 400]
		]])

	points_dst = torch.FloatTensor([[
		[400, 400], [0, 400], [0, 0], [400, 0]
		]])

	M2 = tgm.get_perspective_transform(points_src, points_dst)
	img_warp2 = tgm.warp_perspective(img, M2 @ M1, dsize=(400, 400))

	image_warp = tgm.utils.tensor_to_image(torch.squeeze(img_warp2.byte(), dim=0))

	for i in range(4):
		center = tuple(points_src[0, i].long().numpy())
		image = cv2.circle(image.copy(), center, 5, (0, 255, 0), -1)


	fig, axs = plt.subplots(1, 2, figsize=(16, 10))
	axs = axs.ravel()

	axs[0].axis('off')
	axs[0].set_title('image source')
	axs[0].imshow(image)

	axs[1].axis('off')
	axs[1].set_title('image destination')
	axs[1].imshow(image_warp)

	plt.show()


class Align(nn.Module):
	def __init__(self):
		super(Align, self).__init__()

		points_src = torch.FloatTensor([[
			[190,210],[455,210],[633,475],[0,475],
		]])	
		points_dst = torch.FloatTensor([[
			[0, 0], [399, 0], [399, 399], [0, 399],
		]])
		self.cropH = tgm.get_perspective_transform(points_src, points_dst)

		points_src = torch.FloatTensor([[
			[0, 0], [400, 0], [400, 400], [0, 400]
			]])
		points_dst = torch.FloatTensor([[
			[400, 400], [0, 400], [0, 0], [400, 0]
			]])
		self.flipH = tgm.get_perspective_transform(points_src, points_dst)


	def forward(self, img1, img2):
		img_warp1 = tgm.warp_perspective(img1, self.cropH, dsize=(400, 400))
		img_warp2 = tgm.warp_perspective(img2, self.flipH @ self.cropH, dsize=(400, 400))

		# im1 = tgm.utils.tensor_to_image(torch.squeeze(img_warp1.byte(), dim=0))
		# im2 = tgm.utils.tensor_to_image(torch.squeeze(img_warp2.byte(), dim=0))

		# fig, axs = plt.subplots(1, 2, figsize=(16, 10))
		# axs = axs.ravel()

		# axs[0].axis('off')
		# axs[0].set_title('image source')
		# axs[0].imshow(im1)

		# axs[1].axis('off')
		# axs[1].set_title('image destination')
		# axs[1].imshow(im2)

		# plt.show()

		return img_warp1, img_warp2


if __name__ == '__main__':
	imgFile1 = argv[1]
	imgFile2 = argv[2]

	# rotate(imgFile)

	# warp(imgFile2)

	image1 = cv2.imread(imgFile1)[..., (2,1,0)]
	img1 = tgm.utils.image_to_tensor(image1)
	img1 = torch.unsqueeze(img1.float(), dim=0)

	image2 = cv2.imread(imgFile2)[..., (2,1,0)]
	img2 = tgm.utils.image_to_tensor(image2)
	img2 = torch.unsqueeze(img2.float(), dim=0)

	alignment = Align()
	im1, im2 = alignment(img1, img2)

	print(im1.shape, im2.shape, type(im1), im1.requires_grad)