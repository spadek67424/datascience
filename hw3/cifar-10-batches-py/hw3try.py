from __future__ import print_function
import keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import History
import numpy as np
import os
import sys
import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))
'''
try et model
'''
channels = 3
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict
def unpickle(file):  
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def convert_images(raw_images):
    raw = np.array(raw_images, dtype = float)/255.0
    images = raw.reshape([-1, channels, size, size])
    images = images.transpose([0, 2, 3, 1])
    return images
size =32
def load_data(file):
    data = unpickle(file)
    images_array = data[b'data']
    images = convert_images(images_array)
    labels = np.array(data[b'labels'])
    return images, labels

def get_test_data():
    images, labels = load_data("test_batch")
    return images, labels, np_utils.to_categorical(labels, 10)

def get_train_data():
    images = np.zeros(shape = [50000, size, size, channels], dtype = float)
    labels = np.zeros(shape = [50000], dtype = int)
    start = 0

    for i in range(5):
        images_batch, labels_batch = load_data("data_batch_" + str(i+1))
        end = start + 10000
        images[start:end,:] = images_batch        
        labels[start:end] = labels_batch
        start = end

    return images, labels, np_utils.to_categorical(labels,10)
def norm(nonx):
	nonx=nonx/255

	mean=np.mean(nonx)
	std=np.std(nonx)
	fex=(nonx-mean)/std
	return fex

if __name__=='__main__':
	
	trainx,labelx,trainy=get_train_data()
	testx,labely,testy=get_test_data()




	'''
	model = Sequential()
	model.add(Conv2D(64,(5,5),input_shape=(32,32,3),activation='relu'));
	model.add(MaxPooling2D((2,2),strides=2))
	
	model.add(Flatten())
	model.add(Dense(units=10,activation='softmax'))
	
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.fit(trainx,trainy,validation_split=0,shuffle=True,batch_size=100,epochs=50)
	model.evaluate(testx,testy);
	'''

	'''
	try model
	'''
	

	model = Sequential()

	#model.add(Conv2D(32, (3,3), padding='same', input_shape= x_train.shape[1:]))
	model.add(Conv2D(32, kernel_size=(3, 3), padding='same',input_shape=(32,32,3)))
	model.add(Activation('relu'))  
	model.add(Conv2D(32, (3, 3)))  
	model.add(Activation('relu'))  
	model.add(Dropout(0.25))  
	  
	model.add(Conv2D(64, (3, 3), padding='same'))  
	model.add(Activation('relu'))  
	model.add(Conv2D(64, (3, 3)))  
	model.add(Activation('relu'))  
	model.add(AveragePooling2D(pool_size=(2, 2)))  
	model.add(Dropout(0.25))  
	  
	model.add(Conv2D(128, (3, 3), padding='same'))  
	model.add(Activation('relu'))  
	model.add(Conv2D(128, (3, 3)))  
	model.add(Activation('relu'))  
	model.add(AveragePooling2D(pool_size=(2, 2)))  
	model.add(Dropout(0.25))  
	  
	model.add(Flatten())  
	model.add(Dense(512))  
	model.add(Activation('relu'))  
	model.add(Dropout(0.5))  
	model.add(Dense(10))  
	model.add(Activation('softmax'))  
	

	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

	epochs = 400
	lrate = 0.01
	decay = lrate/epochs
	opt = keras.optimizers.Adam(decay=0.000001)
	#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])
	print(model.summary())
	datagen.fit(trainx)
	his=model.fit_generator(datagen.flow(trainx, trainy,
                                     batch_size=32),
                        epochs=epochs,
                        validation_data=(testx, testy),
                        steps_per_epoch=50000//32)
	
	'''
	model.fit(trainx, trainy,
              batch_size=32,
              epochs=epochs,
              validation_data=(testx, testy),
              shuffle=True)
	'''

	
	# Final evaluation of the model
	scores = model.evaluate(testx, testy, verbose=1)
	model.save('model.h5')
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	plt.plot(his.history['acc'])
	plt.plot(his.history['val_acc'])
	plt.savefig('one.png')
	
	#print(dat.a)
