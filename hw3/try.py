import keras
from keras.datasets import cifar10  
from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
batch_size = 128  
num_classes = 10  
epochs = 1 
data_augmentation = True

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  
print('X_train shape:', x_train.shape)  
print(x_train.shape[0], 'train samples')  
print(x_test.shape[0], 'test samples')  
print(x_train.shape[1:])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
  
model.add(Conv2D(256, (3, 3), padding='same'))  
model.add(Activation('relu'))  
model.add(Conv2D(256, (1, 1)))  
model.add(Activation('relu'))  
model.add(AveragePooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.25))  
  
model.add(Flatten())  
model.add(Dense(512))  
model.add(Activation('relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes))  
model.add(Activation('softmax'))  
  
# initiate RMSprop optimizer  
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  
opt2 = keras.optimizers.adam(decay=1e-6)  
# Let's train the model using RMSprop  
model.compile(loss='categorical_crossentropy',  
              optimizer=opt2,  
              metrics=['accuracy'])  


model.summary()  
x_train = x_train.astype('float32')  
x_test = x_test.astype('float32')  
x_train /= 255  
x_test /= 255  
print(x_train[0].shape)
def sparse(np_array):
	left_right= np.zeros((128,48,3))
	up_down = np.zeros((48,32,3))
	up = np.concatenate((up_down, np_array), axis=0)
	down = np.concatenate((up, up_down), axis=0)
	left = np.concatenate((left_right, down), axis=1)
	right = np.concatenate((left, left_right), axis=1)
	return right
'''
print(x_train.shape[0])
reshape_x_train = np.zeros((50000,128,128,3))
reshape_x_test = np.zeros((10000,128,128,3))
for i in range(x_train.shape[0]):
	reshape_x_train[i] = sparse(x_train[i])
	if i < 10000:
		reshape_x_test[i] = sparse(x_test[i])
	print(i)
print(reshape_x_train.shape)
print(reshape_x_test.shape)
print(reshape_x_train[0])
np.save('reshape_x_train.npy', reshape_x_train)
np.save('reshape_x_test.npy', reshape_x_test)

reshape_x_train = np.load('reshape_x_train.npy')
reshape_x_test = np.load('reshape_x_test.npy')
'''
x_train , x_val, y_train, y_val =  train_test_split(x_train, y_train, test_size=0.1, random_state=0) 

if not data_augmentation:  
    print('Not using data augmentation.')  
    model.fit(x_train, y_train,  
              batch_size=batch_size,  
              epochs=epochs,  
              validation_data=(x_test, y_test),  
              shuffle=True)  
else:
    print('Using real-time data augmentation.')  
    # This will do preprocessing and realtime data augmentation:  
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
  
    # Compute quantities required for feature-wise normalization  
    # (std, mean, and principal components if ZCA whitening is applied).  
    datagen.fit(x_train)  
  
    # Fit the model on the batches generated by datagen.flow().  
    history = model.fit_generator(datagen.flow(x_train, y_train,  
                                     batch_size=batch_size),  
                        steps_per_epoch=x_train.shape[0] // batch_size,  
                        epochs=epochs,  
                        validation_data=(x_val, y_val),
			verbose=2)  
    #model.save('model_avgpooling_sparse_480.h5')

scores = model.evaluate(x_test, y_test, batch_size = batch_size)
print('Test loss : ', scores[0])
print('Test accuracy :', scores[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
plt.savefig('accuracy_avgpooling_480.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_avgpoling_sparse_480.png')'''
