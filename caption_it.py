from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image

def extract_features(filename):
	
	model = VGG16()
	
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model, tokenizer, photo, max_length):
	
	in_text = 'startseq'
	
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text

tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 33
#photo = []
model = load_model('model-ep004-loss3.561-val_loss3.823.h5')	
photo0 = extract_features('woman-in-purple-shirt-walking-her-dog-on-road.jpg')
photo1 = extract_features('iiitb6.jpg')
photo3 = extract_features('5a680cf36687bb.png')
photo5 = extract_features('stpark.jpg')
photo6 = extract_features('iiitb1.jpg')
photo7 = extract_features('shutterstock_59018779.jpg')
photo8 = extract_features('19243026_1506861606055966_4237860359978951616_o.jpg')

description = {}
description[0] = generate_desc(model, tokenizer, photo0, max_length)
description[1] = generate_desc(model, tokenizer, photo1, max_length)

description[2] = generate_desc(model, tokenizer, photo3, max_length)	
description[3] = generate_desc(model, tokenizer, photo5, max_length)
description[4] = generate_desc(model, tokenizer, photo6, max_length)
description[5] = generate_desc(model, tokenizer, photo7, max_length)
description[6] = generate_desc(model, tokenizer, photo8, max_length)

for i in range(7):
	
	print("-----------------------------------------------------------------------")
	print "Image ID:", i+1
	print "Image Description:"
	
	print(description[i])
	
