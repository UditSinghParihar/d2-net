import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchgeometry as tgm

import matplotlib.pyplot as plt
from lib.utils import imshow_image
from sys import exit


class DenseFeatureExtractionModule(nn.Module):
	def __init__(self, finetune_feature_extraction=False, use_cuda=True):
		super(DenseFeatureExtractionModule, self).__init__()

		model = models.vgg16()
		vgg16_layers = [
			'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
			'pool1',
			'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
			'pool2',
			'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
			'pool3',
			'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
			'pool4',
			'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
			'pool5'
		]
		conv4_3_idx = vgg16_layers.index('conv4_3')

		self.model = nn.Sequential(
			*list(model.features.children())[: conv4_3_idx + 1]
		)

		self.num_channels = 512

		# Fix forward parameters
		for param in self.model.parameters():
			param.requires_grad = False
		if finetune_feature_extraction:
			# Unlock conv4_3
			for param in list(self.model.parameters())[-2 :]:
				param.requires_grad = True

		if use_cuda:
			self.model = self.model.cuda()

	def forward(self, batch):
		output = self.model(batch)
		return output


class SoftDetectionModule(nn.Module):
	def __init__(self, soft_local_max_size=3):
		super(SoftDetectionModule, self).__init__()

		self.soft_local_max_size = soft_local_max_size

		self.pad = self.soft_local_max_size // 2

	def forward(self, batch):
		b = batch.size(0)

		batch = F.relu(batch)

		max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
		exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
		sum_exp = (
			self.soft_local_max_size ** 2 *
			F.avg_pool2d(
				F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
				self.soft_local_max_size, stride=1
			)
		)
		local_max_score = exp / (sum_exp )

		depth_wise_max = torch.max(batch, dim=1)[0]
		depth_wise_max_score = batch / (depth_wise_max.unsqueeze(1) )

		all_scores = local_max_score * depth_wise_max_score
		score = torch.max(all_scores, dim=1)[0]

		score = score / (torch.sum(score.view(b, -1), dim=1).view(b, 1, 1) + 1e-5)

		return score


class Align(nn.Module):
	def __init__(self):
		super(Align, self).__init__()

		# Resnet-18 features
		# Predict 8 points (4 pixels) per homography matrix

		points_src = torch.FloatTensor([[
			[190,210],[455,210],[633,475],[0,475],
		]]).cuda()	
		points_dst = torch.FloatTensor([[
			[0, 0], [399, 0], [399, 399], [0, 399],
		]]).cuda()
		cropH = tgm.get_perspective_transform(points_src, points_dst)

		points_src = torch.FloatTensor([[
			[0, 0], [400, 0], [400, 400], [0, 400]
			]]).cuda()
		points_dst = torch.FloatTensor([[
			[400, 400], [0, 400], [0, 0], [400, 0]
			]]).cuda()
		flipH = tgm.get_perspective_transform(points_src, points_dst)

		self.H1 = cropH
		# self.H2 = flipH @ cropH
		self.H2 = cropH


	def forward(self, img1, img2):
		img_warp1 = tgm.warp_perspective(img1, self.H1, dsize=(400, 400))
		img_warp2 = tgm.warp_perspective(img2, self.H2, dsize=(400, 400))

		return img_warp1, img_warp2, self.H1, self.H2
		

class D2Net(nn.Module):
	def __init__(self, model_file=None, use_cuda=True):
		super(D2Net, self).__init__()
		
		self.dense_feature_extraction = DenseFeatureExtractionModule(
			finetune_feature_extraction=True,
			use_cuda=use_cuda
		)

		self.detection = SoftDetectionModule()

		if model_file is not None:
			if use_cuda:
				self.load_state_dict(torch.load(model_file)['model'])
			else:
				self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

	def forward(self, batch):
		b = batch['image1'].size(0)

		dense_features = self.dense_feature_extraction(
			torch.cat([batch['image1'], batch['image2']], dim=0)
		)

		scores = self.detection(dense_features)

		dense_features1 = dense_features[: b, :, :, :]
		dense_features2 = dense_features[b :, :, :, :]

		scores1 = scores[: b, :, :]
		scores2 = scores[b :, :, :]

		return {
			'dense_features1': dense_features1,
			'scores1': scores1,
			'dense_features2': dense_features2,
			'scores2': scores2
		}


class D2NetAlign(nn.Module):
	def __init__(self, model_file=None, use_cuda=True):
		super(D2NetAlign, self).__init__()
		
		self.alignment = Align()

		self.dense_feature_extraction = DenseFeatureExtractionModule(
			finetune_feature_extraction=True,
			use_cuda=use_cuda
		)

		self.detection = SoftDetectionModule()

		if model_file is not None:
			if use_cuda:
				self.load_state_dict(torch.load(model_file)['model'])
			else:
				self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])


	def display(self, img1, img2):
		plt.figure()

		im1 = imshow_image(
			img1[0].cpu().numpy(),
			preprocessing='caffe'
		)

		im2 = imshow_image(
			img2[0].cpu().numpy(),
			preprocessing='caffe'
		)

		plt.subplot(1, 2, 1)
		plt.imshow(im1)
		plt.axis('off')

		plt.subplot(1, 2, 2)
		plt.imshow(im2)
		plt.axis('off')

		plt.show()

		exit(1)


	def forward(self, batch):
		b = batch['image1'].size(0)
		
		# img_warp1, img_warp2, H1, H2 = self.alignment(batch['image1'], batch['image2'])

		# dense_features = self.dense_feature_extraction(
		# 	torch.cat([img_warp1, img_warp2], dim=0)
		# )

		# self.display(img_warp1, img_warp2)

		dense_features = self.dense_feature_extraction(
			torch.cat([batch['image1'], batch['image2']], dim=0)
		)

		scores = self.detection(dense_features)

		dense_features1 = dense_features[: b, :, :, :]
		dense_features2 = dense_features[b :, :, :, :]

		scores1 = scores[: b, :, :]
		scores2 = scores[b :, :, :]

		return {
			'dense_features1': dense_features1,
			'scores1': scores1,
			'dense_features2': dense_features2,
			'scores2': scores2,
			# 'H1': H1,
			# 'H2': H2 
		}