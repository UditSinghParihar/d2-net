import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cv2
from sys import exit

import torch
import torch.nn.functional as F

from lib.utils import (
	grid_positions,
	upscale_positions,
	downscale_positions,
	savefig,
	imshow_image
)
from lib.exceptions import NoGradientError, EmptyTensorError
import torchgeometry as tgm


# matplotlib.use('Agg')


def loss_function(
		model, batch, device, margin=0.3, safe_radius=4, scaling_steps=3, plot=False
):
	output = model({
		'image1': batch['image1'].to(device),
		'image2': batch['image2'].to(device)
	})

	loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
	has_grad = False

	n_valid_samples = 0
	for idx_in_batch in range(batch['image1'].size(0)):
		# Annotations
		depth1 = batch['depth1'][idx_in_batch].to(device)  # [h1, w1]
		intrinsics1 = batch['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
		pose1 = batch['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
		bbox1 = batch['bbox1'][idx_in_batch].to(device)  # [2]

		depth2 = batch['depth2'][idx_in_batch].to(device)
		intrinsics2 = batch['intrinsics2'][idx_in_batch].to(device)
		pose2 = batch['pose2'][idx_in_batch].view(4, 4).to(device)
		bbox2 = batch['bbox2'][idx_in_batch].to(device)

		# Network output
		dense_features1 = output['dense_features1'][idx_in_batch]
		c, h1, w1 = dense_features1.size()
		scores1 = output['scores1'][idx_in_batch].view(-1)

		dense_features2 = output['dense_features2'][idx_in_batch]
		_, h2, w2 = dense_features2.size()
		scores2 = output['scores2'][idx_in_batch]


		all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
		descriptors1 = all_descriptors1

		all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

		# Warp the positions from image 1 to image 2
		fmap_pos1 = grid_positions(h1, w1, device)

		hOrig, wOrig = int(batch['image1'].shape[2]/8), int(batch['image1'].shape[3]/8)
		fmap_pos1Orig = grid_positions(hOrig, wOrig, device)
		pos1 = upscale_positions(fmap_pos1Orig, scaling_steps=scaling_steps)

		# SIFT Feature Detection
		
		imgNp1 = imshow_image(
						batch['image1'][idx_in_batch].cpu().numpy(),
						preprocessing=batch['preprocessing']
					)
		imgNp1 = cv2.cvtColor(imgNp1, cv2.COLOR_BGR2RGB)
		surf = cv2.xfeatures2d.SIFT_create(300)
		# surf = cv2.xfeatures2d.SURF_create(300)
		kp = surf.detect(imgNp1, None)
		keyP = [(kp[i].pt) for i in range(len(kp))]
		keyP = np.asarray(keyP).T
		keyP[[0, 1]] = keyP[[1, 0]]
		keyP = np.floor(keyP) + 0.5

		pos1 = torch.from_numpy(keyP).to(pos1.device).float()
		
		try:
			pos1, pos2, ids = warp(
				pos1,
				depth1, intrinsics1, pose1, bbox1,
				depth2, intrinsics2, pose2, bbox2
			)
		except EmptyTensorError:
			continue

		ids = idsAlign(pos1, device, h1, w1)

		# cv2.drawKeypoints(imgNp1, kp, imgNp1)
		# cv2.imshow('Keypoints', imgNp1)
		# cv2.waitKey(0)

		# drawTraining(batch['image1'], batch['image2'], pos1, pos2, batch, idx_in_batch, output, save=False)
		
		# exit(1)

		# # SIFT Feature Detection
		
		# imgNp1 = imshow_image(
		# 				batch['image1'][idx_in_batch].cpu().numpy(),
		# 				preprocessing=batch['preprocessing']
		# 			)
		# imgNp1 = cv2.cvtColor(imgNp1, cv2.COLOR_BGR2RGB)
		# surf = cv2.xfeatures2d.SURF_create(150)
		# kp = surf.detect(imgNp1, None)
		# keyP = [(kp[i].pt) for i in range(len(kp))]
		# keyP = np.asarray(keyP).T
		# keyP[[0, 1]] = keyP[[1, 0]]
		# keyP = np.floor(keyP) + 0.5

		# pos1, pos2, ids = keyPointCorr(pos1, pos2, ids, keyP)

		# cv2.drawKeypoints(imgNp1, kp, imgNp1)
		# cv2.imshow('Keypoints', imgNp1)
		# cv2.waitKey(0)
		

		# Top view homography adjustment

		# H1 = output['H1'][idx_in_batch] 
		# H2 = output['H2'][idx_in_batch]

		# try:
		# 	pos1, pos2 = homoAlign(pos1, pos2, H1, H2, device)
		# except IndexError:
		# 	continue

		# ids = idsAlign(pos1, device, h1, w1)

		# img_warp1 = tgm.warp_perspective(batch['image1'].to(device), H1, dsize=(400, 400))
		# img_warp2 = tgm.warp_perspective(batch['image2'].to(device), H2, dsize=(400, 400))

		# drawTraining(img_warp1, img_warp2, pos1, pos2, batch, idx_in_batch, output)

		fmap_pos1 = fmap_pos1[:, ids]
		descriptors1 = descriptors1[:, ids]
		scores1 = scores1[ids]

		# Skip the pair if not enough GT correspondences are available
		if ids.size(0) < 128:
			continue

		# Descriptors at the corresponding positions
		fmap_pos2 = torch.round(
			downscale_positions(pos2, scaling_steps=scaling_steps)
		).long()
	
		descriptors2 = F.normalize(
			dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
			dim=0
		)
	
		positive_distance = 2 - 2 * (
			descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
		).squeeze()

		# positive_distance = getPositiveDistance(descriptors1, descriptors2)

		all_fmap_pos2 = grid_positions(h2, w2, device)
		position_distance = torch.max(
			torch.abs(
				fmap_pos2.unsqueeze(2).float() -
				all_fmap_pos2.unsqueeze(1)
			),
			dim=0
		)[0]
		is_out_of_safe_radius = position_distance > safe_radius
		
		distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
		# distance_matrix = getDistanceMatrix(descriptors1, all_descriptors2)

		negative_distance2 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]
		
		# negative_distance2 = semiHardMine(distance_matrix, is_out_of_safe_radius, positive_distance, margin)
		
		all_fmap_pos1 = grid_positions(h1, w1, device)
		position_distance = torch.max(
			torch.abs(
				fmap_pos1.unsqueeze(2).float() -
				all_fmap_pos1.unsqueeze(1)
			),
			dim=0
		)[0]
		is_out_of_safe_radius = position_distance > safe_radius
		
		distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
		# distance_matrix = getDistanceMatrix(descriptors2, all_descriptors1)
		
		negative_distance1 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]

		# negative_distance1 = semiHardMine(distance_matrix, is_out_of_safe_radius, positive_distance, margin)

		diff = positive_distance - torch.min(
			negative_distance1, negative_distance2
		)

		scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

		loss = loss + (
			torch.sum(scores1 * scores2 * F.relu(margin + diff)) /
			(torch.sum(scores1 * scores2) )
		)

		has_grad = True
		n_valid_samples += 1

		if plot and batch['batch_idx'] % batch['log_interval'] == 0:
			drawTraining(batch['image1'], batch['image2'], pos1, pos2, batch, idx_in_batch, output, save=True)
			# drawTraining(img_warp1, img_warp2, pos1, pos2, batch, idx_in_batch, output, save=True)

	if not has_grad:
		raise NoGradientError

	loss = loss / (n_valid_samples )

	return loss


def interpolate_depth(pos, depth):
	# Depth filtering and interpolation of sparse depth

	device = pos.device

	ids = torch.arange(0, pos.size(1), device=device)

	h, w = depth.size()

	i = pos[0, :]
	j = pos[1, :]

	# Valid corners
	i_top_left = torch.floor(i).long()
	j_top_left = torch.floor(j).long()
	valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

	i_top_right = torch.floor(i).long()
	j_top_right = torch.ceil(j).long()
	valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

	i_bottom_left = torch.ceil(i).long()
	j_bottom_left = torch.floor(j).long()
	valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

	i_bottom_right = torch.ceil(i).long()
	j_bottom_right = torch.ceil(j).long()
	valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

	valid_corners = torch.min(
		torch.min(valid_top_left, valid_top_right),
		torch.min(valid_bottom_left, valid_bottom_right)
	)

	i_top_left = i_top_left[valid_corners]
	j_top_left = j_top_left[valid_corners]

	i_top_right = i_top_right[valid_corners]
	j_top_right = j_top_right[valid_corners]

	i_bottom_left = i_bottom_left[valid_corners]
	j_bottom_left = j_bottom_left[valid_corners]

	i_bottom_right = i_bottom_right[valid_corners]
	j_bottom_right = j_bottom_right[valid_corners]

	ids = ids[valid_corners]

	if ids.size(0) == 0:
		raise EmptyTensorError

	# Valid depth
	valid_depth = torch.min(
		torch.min(
			depth[i_top_left, j_top_left] > 0,
			depth[i_top_right, j_top_right] > 0
		),
		torch.min(
			depth[i_bottom_left, j_bottom_left] > 0,
			depth[i_bottom_right, j_bottom_right] > 0
		)
	)

	i_top_left = i_top_left[valid_depth]
	j_top_left = j_top_left[valid_depth]

	i_top_right = i_top_right[valid_depth]
	j_top_right = j_top_right[valid_depth]

	i_bottom_left = i_bottom_left[valid_depth]
	j_bottom_left = j_bottom_left[valid_depth]

	i_bottom_right = i_bottom_right[valid_depth]
	j_bottom_right = j_bottom_right[valid_depth]

	ids = ids[valid_depth]

	if ids.size(0) == 0:
		raise EmptyTensorError

	# Interpolation
	i = i[ids]
	j = j[ids]
	dist_i_top_left = i - i_top_left.float()
	dist_j_top_left = j - j_top_left.float()
	w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
	w_top_right = (1 - dist_i_top_left) * dist_j_top_left
	w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
	w_bottom_right = dist_i_top_left * dist_j_top_left

	interpolated_depth = (
		w_top_left * depth[i_top_left, j_top_left] +
		w_top_right * depth[i_top_right, j_top_right] +
		w_bottom_left * depth[i_bottom_left, j_bottom_left] +
		w_bottom_right * depth[i_bottom_right, j_bottom_right]
	)

	pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

	return [interpolated_depth, pos, ids]


def uv_to_pos(uv):
	return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def warp(
		pos1,
		depth1, intrinsics1, pose1, bbox1,
		depth2, intrinsics2, pose2, bbox2
):
	device = pos1.device

	Z1, pos1, ids = interpolate_depth(pos1, depth1)
	# COLMAP convention
	u1 = pos1[1, :] + bbox1[1] + .5
	v1 = pos1[0, :] + bbox1[0] + .5

	X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
	Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

	XYZ1_hom = torch.cat([
		X1.view(1, -1),
		Y1.view(1, -1),
		Z1.view(1, -1),
		torch.ones(1, Z1.size(0), device=device)
	], dim=0)
	XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
	XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

	uv2_hom = torch.matmul(intrinsics2, XYZ2)
	uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

	u2 = uv2[0, :] - bbox2[1] - .5
	v2 = uv2[1, :] - bbox2[0] - .5
	uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

	annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)

	ids = ids[new_ids]
	pos1 = pos1[:, new_ids]
	estimated_depth = XYZ2[2, new_ids]

	differnce = torch.abs(estimated_depth - annotated_depth)
	inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

	ids = ids[inlier_mask]
	if ids.size(0) == 0:
		raise EmptyTensorError

	pos2 = pos2[:, inlier_mask]
	pos1 = pos1[:, inlier_mask]

	return pos1, pos2, ids


def drawTraining(image1, image2, pos1, pos2, batch, idx_in_batch, output, save=False):
	pos1_aux = pos1.cpu().numpy()
	pos2_aux = pos2.cpu().numpy()

	k = pos1_aux.shape[1]
	col = np.random.rand(k, 3)
	n_sp = 4
	plt.figure()
	plt.subplot(1, n_sp, 1)
	im1 = imshow_image(
		image1[0].cpu().numpy(),
		preprocessing=batch['preprocessing']
	)
	plt.imshow(im1)
	plt.scatter(
		pos1_aux[1, :], pos1_aux[0, :],
		s=0.25**2, c=col, marker=',', alpha=0.5
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 2)
	plt.imshow(
		output['scores1'][idx_in_batch].data.cpu().numpy(),
		cmap='Reds'
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 3)
	im2 = imshow_image(
		image2[0].cpu().numpy(),
		preprocessing=batch['preprocessing']
	)
	plt.imshow(im2)
	plt.scatter(
		pos2_aux[1, :], pos2_aux[0, :],
		s=0.25**2, c=col, marker=',', alpha=0.5
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 4)
	plt.imshow(
		output['scores2'][idx_in_batch].data.cpu().numpy(),
		cmap='Reds'
	)
	plt.axis('off')

	if(save == True):
		savefig('train_vis/%s.%02d.%02d.%d.png' % (
			'train' if batch['train'] else 'valid',
			batch['epoch_idx'],
			batch['batch_idx'] // batch['log_interval'],
			idx_in_batch
		), dpi=300)
	else:
		plt.show()
	
	plt.close()

	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

	for i in range(0, pos1_aux.shape[1], 1):
		im1 = cv2.circle(im1, (pos1_aux[1, i], pos1_aux[0, i]), 1, (0, 0, 255), 2)
	for i in range(0, pos2_aux.shape[1], 1):
		im2 = cv2.circle(im2, (pos2_aux[1, i], pos2_aux[0, i]), 1, (0, 0, 255), 2)

	im3 = cv2.hconcat([im1, im2])

	for i in range(0, pos1_aux.shape[1], 1):
		im3 = cv2.line(im3, (int(pos1_aux[1, i]), int(pos1_aux[0, i])), (int(pos2_aux[1, i]) +  im1.shape[1], int(pos2_aux[0, i])), (0, 255, 0), 1)

	if(save == True):
		cv2.imwrite('train_vis/%s.%02d.%02d.%d.png' % (
			'train_corr' if batch['train'] else 'valid',
			batch['epoch_idx'],
			batch['batch_idx'] // batch['log_interval'],
			idx_in_batch
		), im3)
	else:
		cv2.imshow('Image', im3)
		cv2.waitKey(0)


def homoAlign(pos1, pos2, H1, H2, device):
	ones = torch.ones(pos1.shape[1]).reshape(1, pos1.shape[1]).to(device)

	pos1[[0, 1]] = pos1[[1, 0]]
	pos2[[0, 1]] = pos2[[1, 0]]

	pos1Homo = torch.cat((pos1, ones), dim=0)
	pos2Homo = torch.cat((pos2, ones), dim=0)

	pos1Warp = H1 @ pos1Homo
	pos2Warp = H2 @ pos2Homo

	pos1Warp = pos1Warp/pos1Warp[2, :]
	pos1Warp = pos1Warp[0:2, :]

	pos2Warp = pos2Warp/pos2Warp[2, :]
	pos2Warp = pos2Warp[0:2, :]

	pos1Warp[[0, 1]] = pos1Warp[[1, 0]]
	pos2Warp[[0, 1]] = pos2Warp[[1, 0]]

	pos1Pov = []
	pos2Pov = []

	for i in range(pos1.shape[1]):
		if(380 > pos1Warp[0, i] > 0 and 380 > pos1Warp[1, i] > 0 and 380 > pos2Warp[0, i] > 0 and 380 > pos2Warp[1, i] > 0):
			pos1Pov.append((pos1Warp[0, i], pos1Warp[1, i]))
			pos2Pov.append((pos2Warp[0, i], pos2Warp[1, i]))

	pos1Pov = torch.Tensor(pos1Pov).to(device)
	pos2Pov = torch.Tensor(pos2Pov).to(device)

	pos1Pov = torch.transpose(pos1Pov, 0, 1)
	pos2Pov = torch.transpose(pos2Pov, 0, 1)

	return pos1Pov, pos2Pov


def idsAlign(pos1, device, h1, w1):
	row = pos1[0, :]/8
	col = pos1[1, :]/8

	ids = []

	for i in range(row.shape[0]):
		index = (h1 * row[i]) + col[i]
		ids.append(index)

	ids = torch.round(torch.Tensor(ids)).long()

	return ids


def semiHardMine(distance_matrix, is_out_of_safe_radius, positive_distance, margin):
	negative_distances = distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.
	
	negDist = []

	for i, row in enumerate(negative_distances):
		posDist = positive_distance[i]
		
		row = row[(posDist + margin > row) & (row > posDist)]
		
		if(row.size(0) == 0):
			negDist.append(negative_distances[i, 0])
		else:
			perm = torch.randperm(row.size(0))
			negDist.append(row[perm[0]])
		
	negDist = torch.Tensor(negDist).to(positive_distance.device)

	return negDist


def getPositiveDistance(descriptors1, descriptors2):
	positive_distance = torch.norm(descriptors1 - descriptors2, dim=0)

	return positive_distance


def getDistanceMatrix(descriptors1, all_descriptors2):
	d1 = descriptors1.t().unsqueeze(0)
	all_d2 = all_descriptors2.t().unsqueeze(0)
	distance_matrix = torch.cdist(d1, all_d2, p=2).squeeze()

	return distance_matrix


# def keyPointCorr(pos1, pos2, ids, keyP):
# 	keyP = torch.from_numpy(keyP).to(pos1.device)
# 	# print("Keypoint: ", keyP.shape)
# 	print("Pos1: {} Pos2: {} Id: {}".format(pos1.shape, pos2.shape, ids.shape))
# 	# print(pos1[0, 0], pos1[1, 0])

# 	print(torch.unique(pos1[0, :]))
# 	print(torch.unique(pos1[1, :]))
# 	# print(keyP[0, 0], keyP[1, 0])

# 	newIds = []
# 	for col in range(keyP.shape[1]):
# 		pass
# 		# if((keyP[0, col] in pos1[0, :]) and (keyP[1, col] in pos1[1, :])):
# 		# 	print("True", col)
# 		# 	newIds.append(col)

# 	return None, None, None

