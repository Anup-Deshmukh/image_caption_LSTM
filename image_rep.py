from os import listdir
from pickle import dump
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


import time
start_time = time.time()
def extract_features(directory):
	model = InceptionV3()
	model.layers.pop()
	
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	#print(model.summary())
	
	features = dict()
	for name in listdir(directory):
		#name = listdir(directory)[0]	
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		#print image.shape

		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		#print image.shape

		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		#print feature
		#print feature.shape

		image_id = name.split('.')[0]
		features[image_id] = feature
		#print('>%s' % name)
	return features

directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print("--- %s seconds ---" % (time.time() - start_time))
print('Extracted Features: %d' % len(features))
dump(features, open('features_inception_flick.pkl', 'wb'))