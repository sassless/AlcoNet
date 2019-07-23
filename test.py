from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys
from decoder import decode_and_show


# Testing params:
model_name = 'AlcoNet_trained100.h5' # path to trained model
# Default paths:
folder_path = './test/'
image_path = './test.jpg'

def check_model(model_name):
	global all_is_ok
	print('loading model: ', end='')
	if os.path.exists(model_name):
		model = load_model(model_name)
		print('done.')
	else:
		print('wrong model file.')
		all_is_ok = False
	return model

def check_mode():
	global all_is_ok
	try:
		mode = sys.argv[1]
	except IndexError:
		print('wrong parameters')
		mode = None
	if not mode in ['0','1']:
		print('wrong mode.')
		all_is_ok = False
	return mode

def check_path():
	global all_is_ok
	global folder_path
	global image_path
	try:
		path = sys.argv[2]
	except IndexError:
		path = None
	if path == None:
		if mode == '0':
			path = folder_path
		elif mode == '1':
			path = image_path
	if not os.path.exists(path):
		print('wrong path [{}]'.format(path))
		all_is_ok = False
	return path

def main(mode, path):
	if mode == '0':
	# Testing on folder with pictures -->
		for file in os.listdir(path):
			image_path = path + file
			img = image.load_img(image_path, target_size=(224,224))
			img_array = image.img_to_array(img)
			x = np.expand_dims(img_array, axis=0)
			x = preprocess_input(x)
			prediction = model.predict(x)
			decode_and_show(img_array,prediction[0])
	elif mode == '1':
	# Testing on unit picture -->	
		img = image.load_img(path, target_size=(224,224))
		img_array = image.img_to_array(img)
		x = np.expand_dims(img_array, axis=0)
		x = preprocess_input(x)
		prediction = model.predict(x)
		decode_and_show(img_array,prediction[0])

all_is_ok = True
model = check_model(model_name)
if all_is_ok:
	mode = check_mode()
if all_is_ok:
	path = check_path()
if all_is_ok:
	main(mode,path)
