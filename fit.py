from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint


# Set training params:
model_name = 'AlcoNet.h5' # path to not trained model
epochs = 100
batch_size = 64 # orig paper trained all networks with batch_size = 128
steps_per_epoch = 34 # typically should be equal to ceil(number_of_samples / batch_size)
img_size = (224,224) # size of input images (H,W) | ResNet standart size is 224x224
# Paths to dirs with train and validation datasets:
train_path = './data/train/'
validation_path = './data/validation/'
# Saving path for checkpoints (after each epoch):
save_path = './AlcoNet_trained{epoch:03d}.h5'
# Saving path for final trained model (after all epochs):
final_save_path = 'AlcoNet_trained{}_final.h5'.format(epochs)


# Generator for train data -->

train_datagen = image.ImageDataGenerator(
# Rescales and preprocess the images
# Transforms the images to increase dataset
# Use keras documentation to tune
	rescale = 1./255,
	rotation_range = 25,
	zoom_range = 0.1,
	shear_range = 0.2,
	fill_mode = 'constant',
	cval = 255,
	preprocessing_function = preprocess_input )

train_generator = train_datagen.flow_from_directory(
# Generates train data from selected dir (train_path) for each epoch
# Detects classes by number of folders in selected dir
	train_path,
	target_size = img_size,
	batch_size = batch_size,
	class_mode = 'categorical',
	shuffle = True )

# Generator for validation data -->

test_datagen = image.ImageDataGenerator(
	rescale = 1./255,
	preprocessing_function = preprocess_input )

test_generator = test_datagen.flow_from_directory(
	validation_path,
	target_size = img_size,
	batch_size = batch_size,
	class_mode = 'categorical',
	shuffle = True )

# Checkpoints callback -->

checkpoint = ModelCheckpoint(
	filepath = save_path,
	monitor = 'val_acc',
	verbose = 1,
	save_best_only = True, # if True saving only when val_acc increase
	mode = 'max' )

callbacks = [checkpoint]

# Model training -->

model = load_model(model_name)
model.fit_generator(
	train_generator,
	steps_per_epoch = steps_per_epoch,
	epochs = epochs,
	verbose = 1,
	callbacks = callbacks,
	validation_data = test_generator )

# Evaluate trained model -->
"""
test = model.evaluate_generator(test_generator, verbose=1)
print('Test loss:', test[0])
print('Test accuracy:', test[1])
"""

model.save(final_save_path)
