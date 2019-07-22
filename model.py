from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


# Set model params:
classes = 11
lr = 1e-3
optimizer = Adam(lr=lr)
img_size = (224,224) # size of input images (H,W) | ResNet standart size is 224x224
frozen_layers = 146 # of 175 ResNet50v1 layers frozen for fitting
save_path = 'AlcoNet.h5'


# Loading ResNet50v1 model with pre-trained weights -->
resnet_model = ResNet50(
	include_top = False,
	weights = 'imagenet',
	input_tensor = None,
	input_shape = (*img_size,3) )

# Custom classificator top-model -->
x = GlobalAveragePooling2D(name='avg_pool')(resnet_model.output)
x = Flatten(name='flatten')(x)
x = Dense(256, activation='relu', name='dense')(x)
x = Dropout(0.5, name='dropout')(x)
top_model = Dense(classes, activation='softmax', name='output')(x)

# Final model of alcohol classificator -->
model = Model(inputs=resnet_model.input, outputs=top_model, name='AlcoNet')
for layer in model.layers[:frozen_layers]:
	layer.trainable = False
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print model summary:
print('Layers:', len(model.layers))
model.summary()

model.save(save_path)
