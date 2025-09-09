import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler, BatchSampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_nan(tensor, name=""):
	if torch.isnan(tensor).any():
		print(name, "nan detected")
		return True
	else:
		return False

class PriSecSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.
	one sequential and another few random

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, data_source, num_secondary=1):
		self.data_source = data_source
		self.num_secondary = num_secondary

	def __iter__(self):
		primary_idx = list(range(len(self.data_source)))
		seed = int(torch.empty((), dtype=torch.int64).random_().item())
		generator = torch.Generator()
		generator.manual_seed(seed)
		indices = [torch.tensor(primary_idx)]
		for i in range(self.num_secondary):
			indices.append(torch.randperm(len(self.data_source), generator=generator))
		self.indices = torch.stack(indices, dim = 1).view(-1).tolist()
		# print("sampler: ",self.indices)
		yield from self.indices

	def __len__(self):
		return len(self.data_source)

def collate_fn_PriSec_train_mutual_rand_merge(batch):
	# designed for fake batchsize=2
    # print("hi")
    # fg_num, normal_num = 100, 412
    # fg_num = np.clip(int(512 * np.random.rand()), a_min=50, a_max=450)
	fg_num = np.random.randint(120, 350)
	# # fg_num = 256
	normal_num = 512 - fg_num
	# fg_num = 510
	# normal_num = 2
	# print("resample ratio: ", fg_num/normal_num)

	main_element = batch[0]
	label = main_element[1]
	label_dict = {0:"luad", 1:"lusc"}
	# fg_all = main_element[0][label_dict[label]]
	fg_all = [i for helper in batch for i in helper[0][label_dict[label]]]
	normal_all = [i for helper in batch for i in helper[0]["normal"]]
	fg_select = list(np.random.choice(fg_all, fg_num, replace=False))
	normal_select = list(np.random.choice(normal_all, normal_num, replace=False))
	patches_list = fg_select + normal_select
	patches = torch.stack([torch.from_numpy((torch.load(i))) for i in patches_list])
	return patches, torch.tensor([label], dtype=torch.long)

def collate_fn_PriSec_train_self_merge(batch):
	# designed for fake batchsize=2
	main_element = batch[0]
	label = main_element[1]
	##
	label_dict = {0:"luad", 1:"lusc"}
	fg_num = len(main_element[0][label_dict[label]])
	normal_num = len(main_element[0]["normal"])
	# print("resample ratio: ", fg_num/normal_num)
	##
	patches_all = [i for j in main_element[0].values() for i in j]
	patches_list = list(np.random.choice(patches_all, 512, replace=False))
	patches = torch.stack([torch.from_numpy((torch.load(i))) for i in patches_list])
	return patches, torch.tensor([label], dtype=torch.long)

def collate_fn_PriSec_val(batch):
    # designed for batchsize=1
    # print("hi")
    # patches_list = [item[0]["patches"] for item in batch]
    patches_list = batch[0][0]["patches"]
    label = torch.LongTensor([item[1] for item in batch])
    patches = torch.stack([torch.from_numpy((torch.load(i))) for i in patches_list])
    return patches, label

def collate_fn_batch_patches(batch):
	fg_num = np.random.randint(120, 350)
	normal_num = 512 - fg_num
	img_list = []
	for item in batch:
		fake_slide = item[0]
		fg_select = list(np.random.choice(fake_slide['fg'], fg_num, replace=False))
		normal_select = list(np.random.choice(fake_slide['normal'], normal_num, replace=False))
		patches_list = fg_select + normal_select
		patches = torch.stack([torch.from_numpy((torch.load(i))) for i in patches_list])
		img_list.append(patches)
	img = torch.stack(img_list)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

#NOTE - Adjust the related proportion fg:normal
def collate_fn_preload_batch(batch):
	# fg_proportion = np.random.normal(0.5, 0.1)
	# fg_proportion = np.clip(fg_proportion, 0.05, 0.95)
	# fg_proportion = np.random.beta(2,8)
	# fg_proportion = 0.2
	# fg_proportion = np.random.choice([0.2, 0.5, 0.8])
	fg_proportion = 0.9
	# fg_num = np.random.randint(120, 350)
	fg_num = int(512 * fg_proportion)
	# fg_num = 100
	normal_num = 512 - fg_num
	img_list = []
	for item in batch:
		fake_slide = item[0]
		fg_select = fake_slide['fg'][:fg_num]
		normal_select = fake_slide['normal'][:normal_num]
		patches = torch.cat([fg_select, normal_select], dim = 0)
		img_list.append(patches)
	img = torch.stack(img_list)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	# img = torch.cat([item[0] for item in batch], dim = 0)
	if type(batch[0][0]) == tuple:
		img = torch.stack([item[0][0] for item in batch])
		coords = torch.stack([item[0][1] for item in batch])
		if coords.shape[0] == 1:
			coords = coords.squeeze(0)
	else:
		img = torch.stack([item[0] for item in batch])
	if img.shape[0] == 1:
		img = img.squeeze(0)
	# label = torch.LongTensor([item[1] for item in batch])
	label = torch.LongTensor(np.array([item[1] for item in batch]))
	if type(batch[0][0]) == tuple:
		return [(img, coords), label]
	return [img, label]

def collate_MIL_vila(batch): # img, img, label
	img = torch.stack([item[0] for item in batch])
	if img.shape[0] == 1:
		img = img.squeeze(0)
	img_l = torch.stack([item[1] for item in batch])
	if img_l.shape[0] == 1:
		img_l = img_l.squeeze(0)
	label = torch.LongTensor(np.array([item[2] for item in batch]))
	return [img, img_l, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, batchsize=1, vila=False):
	"""
		return either the validation loader or training loader 
	"""
	if vila:
		collate_fn = collate_MIL_vila
	else:
		collate_fn = collate_MIL
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=batchsize, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, **kwargs )

	return loader

def get_pseudo_loader_pri_sec_slide(split_dataset, training = False, testing = False, weighted = False, num_secondary=1):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	assert num_secondary < len(split_dataset)
	if not testing:
		if training:
			# loader = DataLoader(split_dataset, batch_size=num_secondary+1, sampler = PriSecSampler(split_dataset, num_secondary), collate_fn = collate_fn_PriSec_train_mutual_rand_merge, **kwargs)
			loader = DataLoader(split_dataset, batch_size=num_secondary+1, sampler = PriSecSampler(split_dataset, num_secondary), collate_fn = collate_fn_PriSec_train_self_merge, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs)

	return loader

def get_pseudo_loader_rand_slide(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_fn_batch_patches, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs)

	return loader

def get_pseudo_loader_preload(split_dataset, training = False, testing = False, weighted = False, batchsize=1):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			loader = DataLoader(split_dataset, batch_size=batchsize, sampler = RandomSampler(split_dataset), collate_fn = collate_fn_preload_batch, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs)

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	elif args.opt == "adamW":
		optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg, betas=(0.9, 0.999))
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	# indices = np.arange(samples).astype(int)

	np.random.seed(seed)
	for i in range(n_splits):
		indices = np.arange(samples).astype(int)
		if custom_test_ids is not None:
			indices = np.setdiff1d(indices, custom_test_ids[i])
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids[i])

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids

def generate_split_few(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None, shot=1):	
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			# print(len(possible_indices), test_num[c])
			# val_ids = np.random.choice(possible_indices, test_num[c], replace = False) # validation ids
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)
			
			# train_ids = np.random.choice(remaining_ids, val_num[c], replace=False)
			train_ids = np.random.choice(remaining_ids, shot, replace=False)
			sampled_train_ids.extend(train_ids)

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))
	# print('###### Testing ######')
	# print(len(dataset.slide_cls_ids))
	# for c in range(len(dataset.slide_cls_ids)):
	# 	print(len(dataset.slide_cls_ids[c]))
	# print('###### End Testing ######')
	# print([len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))])
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

