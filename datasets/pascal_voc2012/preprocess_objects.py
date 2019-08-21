import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from optparse import OptionParser

def mkdir(dir_path, dir_name, forced_remove=False):
	new_dir = '{}/{}'.format(dir_path,dir_name)
	if forced_remove and os.path.isdir( new_dir ):
		shutil.rmtree( new_dir )
	if not os.path.isdir( new_dir ):
		os.makedirs( new_dir )

def touch(file_path, file_name, forced_remove=False):
	new_file = '{}/{}'.format(file_path,file_name)
	assert os.path.isdir( file_path ), ' \"{}\" does not exist.'.format(file_path)
	if forced_remove and os.path.isfile(new_file):
		os.remove(new_file)
	if not os.path.isfile(new_file):
		open(new_file, 'a').close()

def write_file(file_path, file_name, content, forced_remove=True):
	touch(file_path, file_name, forced_remove=forced_remove)
	with open('{}/{}'.format(file_path, file_name), 'a') as f:
		f.write('{}\n'.format(content))

def copy_file(src_path, src_file_name, dst_path, dst_file_name):
	shutil.copyfile('{}/{}'.format(src_path, src_file_name), '{}/{}'.format(dst_path,dst_file_name))

def ls(dir_path):
	return os.listdir(dir_path)

def does_exist(dir_path,file_name):
	return os.path.isfile('{}/{}'.format(dir_path, file_name))

def get_colors_area(file_path, file_name):
	background_color = 0
	void_color = 255
	image = Image.open('{}/{}'.format(file_path, file_name))
	colors = image.getcolors()
	colors.sort(key=lambda tup: tup[0])
	areas = [color[0] for color in colors]
	colors = [color[1] for color in colors]
	return colors, areas

def get_labels():
	return ['background', 
			'aeroplane', 
			'bicycle', 
			'bird', 
			'boat', 
			'bottle', 
			'bus', 
			'car', 
			'cat', 
			'chair', 
			'cow', 
			'diningtable', 
			'dog', 
			'horse', 
			'motorbike', 
			'person', 
			'pottedplant', 
			'sheep', 
			'sofa', 
			'train', 
			'tvmonitor', 
			'void']

def save_data(data_mode, pascal_images_dir, pascal_masks_dir, saving_dst='.', saving_point=(0., 1.)):
	assert saving_point[0]<=saving_point[1] and saving_point[0]>=0 and saving_point[1]<=1, 'saving point error: 0<={}<={}<=1 ?!'.format( saving_point[0], saving_point[1])
	images_name = ls('{}'.format(pascal_images_dir))

	starting_point = saving_point[0] * len(images_name)
	ending_point = saving_point[1] * len(images_name)
	num_data = ending_point - starting_point
	
	for ix, (image_name) in enumerate(images_name):
		if ix < starting_point:
			continue
		if ix > ending_point:
			break
		mask_name = '{}.png'.format( os.path.splitext(image_name)[0] )
		if not does_exist( pascal_masks_dir, mask_name ):
			touch(saving_dst, 'mask_does_not_exist.txt', forced_remove=False)
			write_file(saving_dst, 'mask_does_not_exist.txt', image_name)
			continue
		
		colors, areas = get_colors_area(pascal_masks_dir, mask_name)
		num_objs = len(colors) - 2

		curr_data_saving_path ='{}/{}'.format(saving_dst, num_objs)

		mkdir(saving_dst, '{}'.format(num_objs), forced_remove=False)
		mkdir('{}'.format(curr_data_saving_path), 'images', forced_remove=False)
		mkdir('{}'.format(curr_data_saving_path), 'masks', forced_remove=False)
		mkdir('{}'.format(curr_data_saving_path), 'colors', forced_remove=False)


		copy_file(	pascal_images_dir, 
					image_name, 
					'{}/images'.format(curr_data_saving_path), 
					image_name )

		copy_file(	pascal_masks_dir, 
					mask_name, 
					'{}/masks'.format(curr_data_saving_path), 
					mask_name)

		totall_area = sum(areas)
		for (color,area) in zip(colors, areas):
			write_file(	'{}/objs_info'.format(curr_data_saving_path), 
						'{}.txt'.format( os.path.splitext(image_name)[0] ), 
						'{} {} {}'.format(color, area/totall_area, area), 
						forced_remove=False
						)

		print('%s: %d/%d( %.2f %% )' % (data_mode, 
										ix, 
										num_data,
										ix / num_data * 100
										), end='\r')

	labels, _ = get_labels()
	for ix,(label) in enumerate(labels):
		write_file(	saving_dst, 
					'objects_colors.txt', '{} {}'.format(	ix, 
															label), 
					forced_remove=False)
	
	print()

def preprocess(data_dir, data_mode, saving_points, saving_dsts):

	assert data_mode in ['trainval', 'test'], 'Unknown data_mode. data_mode must be one of [\'trainval\', \'test\']'
	pascal_images_dir = '{}/JPEGImages'.format(data_dir)
	pascal_masks_dir = '{}/SegmentationClass'.format(data_dir)

	data_modes = ['train', 'val'] if data_mode=='trainval' else ['test']
	assert len(data_mode)==len(saving_points), 'saving points must has {} saving point.'.format( len(data_mode) )
	assert len(data_mode)==len(saving_dsts), 'saving destinations must has {} saving destination.'.format( len(data_mode) )
	

	
	for (saving_dst, data_mode) in zip(saving_dsts, data_modes):
		mkdir(saving_dst, data_mode)
	
	for (saving_dst, data_mode, saving_point) in zip(saving_dsts, data_modes, saving_points):
		save_data(data_mode, pascal_images_dir, pascal_masks_dir, saving_dst='{}/{}'.format(saving_dst, data_mode), saving_point=saving_point)

def _main(args):
	pascal_trainval_path = args.trainval
	pascal_test_path = args.test
	
	trainval_saving_points = [(0.,0.8), (0.8,1.)]
	test_saving_points = [(0.,1.)]

	trainval_saving_dst = ['.', '.']
	test_saving_dst = '.'

	preprocess(pascal_trainval_path, 'trainval', trainval_saving_points, trainval_saving_dst)
	preprocess(pascal_test_path, 'test', test_saving_points, test_saving_dst)

def get_args():
	parser = OptionParser()
	parser.add_option('--trainval_data', dest='trainval', default='./voc_trainval', type='string',
						help='trainval data path')
	parser.add_option('--test_data', dest='test', default='./voc_test', type='string',
						help='test data path')
	(options, args) = parser.parse_args()
	return options

if __name__ == "__main__":
	args = get_args()
	_main(args)
	
