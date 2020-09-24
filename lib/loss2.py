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

# matplotlib.use('Agg')


def loss_function(
		model, batch, device, margin=1, safe_radius=4, scaling_steps=3, plot=False
):
	output = model({
		'image1': batch['image1'].to(device),
		'image2': batch['image2'].to(device)
	})

	# print(output['dense_features1'].shape, output['dense_features2'].shape, 
	# 	output['scores1'].shape, output['scores2'].shape)
	# print(batch['image1'].shape, batch['image2'].shape)
	# exit(1)

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

		H1 = output['H1'][idx_in_batch] 
		H2 = output['H2'][idx_in_batch]

		all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
		descriptors1 = all_descriptors1
		# print("Descriptors: ", descriptors1.shape)

		all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

		# Warp the positions from image 1 to image 2
		hOrig, wOrig = 60, 80
		fmap_pos1 = grid_positions(hOrig, wOrig, device)

		# fmap_pos1 = grid_positions(h1, w1, device)

		pos1 = upscale_positions(fmap_pos1, scaling_steps=scaling_steps)

		try:
			pos1, pos2, ids = warp(
				pos1,
				depth1, intrinsics1, pose1, bbox1,
				depth2, intrinsics2, pose2, bbox2
			)
		except EmptyTensorError:
			continue

		pos1, pos2, ids, fmap_pos1 = homoAlign(pos1, pos2, ids, fmap_pos1, H1, H2)

		print("Warp output: ", pos1.shape, pos2.shape, ids.shape)
		exit(1)

		fmap_pos1 = fmap_pos1[:, ids]
		descriptors1 = descriptors1[:, ids]
		scores1 = scores1[ids]

		# Skip the pair if not enough GT correspondences are available
		# print("correspondences number: {}".format(ids.size(0)))
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
		# print("Descriptors: ", descriptors1.shape, descriptors2.shape)
		# exit(1)
		positive_distance = 2 - 2 * (
			descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
		).squeeze()

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
		negative_distance2 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]

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
		negative_distance1 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]

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

		log_correspond = 10 
		if plot and batch['batch_idx'] % batch['log_interval'] == 0:
		# if plot and batch['batch_idx'] % log_correspond == 0:
			pos1_aux = pos1.cpu().numpy()
			pos2_aux = pos2.cpu().numpy()
			k = pos1_aux.shape[1]
			col = np.random.rand(k, 3)
			n_sp = 4
			plt.figure()
			plt.subplot(1, n_sp, 1)
			im1 = imshow_image(
				batch['image1'][idx_in_batch].cpu().numpy(),
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
				batch['image2'][idx_in_batch].cpu().numpy(),
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
			savefig('train_vis/%s.%02d.%02d.%d.png' % (
				'train' if batch['train'] else 'valid',
				batch['epoch_idx'],
				# batch['batch_idx'] // batch['log_interval'],
				batch['batch_idx'] // log_correspond,
				idx_in_batch
			), dpi=300)
			plt.close()

			# Plotting correspondences

			# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
			# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
			
			# for i in range(0, pos1_aux.shape[1], 5):
			# 	im1 = cv2.circle(im1, (pos1_aux[1, i], pos1_aux[0, i]), 1, (0, 0, 255), 2)
			# for i in range(0, pos2_aux.shape[1], 5):
			# 	im2 = cv2.circle(im2, (pos2_aux[1, i], pos2_aux[0, i]), 1, (0, 0, 255), 2)
			
			# im3 = cv2.hconcat([im1, im2])

			# for i in range(0, pos1_aux.shape[1], 5):
			# 	im3 = cv2.line(im3, (int(pos1_aux[1, i]), int(pos1_aux[0, i])), (int(pos2_aux[1, i]) +  im1.shape[1], int(pos2_aux[0, i])), (0, 255, 0), 2)

			# cv2.imwrite('train_vis/%s.%02d.%02d.%d.png' % (
			# 	'train_corr' if batch['train'] else 'valid',
			# 	batch['epoch_idx'],
			# 	# batch['batch_idx'] // batch['log_interval'],
			# 	batch['batch_idx'] // log_correspond,
			# 	idx_in_batch
			# ), im3)

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
		# print("EmptyTensorError exception.")
		raise EmptyTensorError

	pos2 = pos2[:, inlier_mask]
	pos1 = pos1[:, inlier_mask]

	return pos1, pos2, ids


def homoAlign(pos1, pos2, ids, fmap_pos1, H1, H2):
	return None, None, None, None