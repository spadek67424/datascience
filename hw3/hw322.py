from __future__ import print_function
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import History
import numpy as np
import os
import sys
import pickle
'''
try imageprocessor
'''

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict
def norm(nonx):
	nonx=nonx/255
	mean=np.mean(nonx)
	std=np.std(nonx)
	fex=(nonx-mean)/std
	return fex

if __name__=='__main__':
	data=list()
	label=list()
	'''
	b'data'
	b'labels'
	b'filenames'
	b'batch_label'

	for i in range()
	'''
	for i in range(1,len(sys.argv)-1):
		dicts=unpickle(sys.argv[i])
		#print(dicts.keys())
		data.append(dicts[b'data'])
		label.append(dicts[b'labels'])
		'''
		print(dicts[b'data'])
		input()
		print(dicts[b'labels'])
		input()
		'''
	trainx=np.array(data)
	trainy=np.array(label)	
	dicts=unpickle(sys.argv[6])
	testx=dicts[b'data']
	testy=dicts[b'labels']
	testy=np.array(testy)
	
	trainx=norm(trainx.reshape((50000,3072)))
	testx=norm(testx.reshape((10000,3072)))
	'''
	trainx=trainx/255
	testx=testx/255
	'''

	trainx=trainx.reshape((50000,32,32,3))
	trainy=trainy.reshape(50000,1)
	trainy=np_utils.to_categorical(trainy,10)
	testx=testx.reshape((10000,32,32,3))
	testy=testy.reshape(10000,1)
	testy=np_utils.to_categorical(testy,10)
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
	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=trainx.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(96, (3, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(96, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
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

	epochs = 1
	lrate = 0.01
	decay = lrate/epochs
	opt = keras.optimizers.Adam(lr=0.0001)
	#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

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
