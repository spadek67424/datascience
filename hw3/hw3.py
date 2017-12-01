from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.utils import np_utils
import numpy as np
import os
import sys
import pickle
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
	for i in range(1,len(sys.argv)):
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
	trainx=norm(trainx.reshape((60000,3072)))
	testx=norm(testx.reshape(10000,3072))
	trainx=trainx.reshape((60000,32,32,3))
	trainy=trainy.reshape(60000,1)
	trainy=np_utils.to_categorical(trainy)
	testx=testx.reshape((10000,32,32,3))
	testy=testy.reshape(10000,1)
	test=np_utils.to_categorical(testy)
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
	model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	epochs = 25
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	print(model.summary())

	model.fit(trainx, trainy, validation_data=(testx, testy), epochs=epochs, batch_size=32)
	# Final evaluation of the model
	scores = model.evaluate(testx, testy, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	#print(dat.a)
