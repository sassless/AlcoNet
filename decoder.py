from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image


classes = [
	'absinthe',
	'beer (bottle)',
	'beer (can)',
	'pink champagne',
	'white champagne',
	'tequila',
	'whiskey',
	'vodka',
	'pink wine',
	'red wine',
	'white wine' ]

def decode_and_show(img_array, prediction):
	global classes
	label = classes[np.argmax(prediction)]
	img = image.array_to_img(img_array)
	#print(prediction)
	plot.title(label)
	plot.imshow(img)
	plot.show()
	