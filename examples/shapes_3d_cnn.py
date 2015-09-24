__author__ = 'Minhaz Palasara'


from keras.datasets import shapes_3d
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

# Data Generation settings
test_split = 0.2
dataset_size = 500
patch_size = 32


(X_train, Y_train),(X_test, Y_test) = shapes_3d.load_data(test_split=test_split,
                                                          dataset_size=dataset_size,
                                                          patch_size=patch_size)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Training Setting
batch_size = 32
nb_classes = 2
nb_epoch = 1

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# number of convolutional filters to use at each layer
nb_filters = [16, 32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2, 2]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [7, 5, 3]

model = Sequential()
model.add(Convolution3D(nb_filters[0], stack_size=1, nb_row=nb_conv[0], nb_col=nb_conv[0], nb_depth=nb_conv[0], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Convolution3D(nb_filters[1], stack_size=nb_filters[0], nb_row=nb_conv[1], nb_col=nb_conv[1], nb_depth=nb_conv[1], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool[1], nb_pool[1], nb_pool[1])))
model.add(Convolution3D(nb_filters[2], stack_size=nb_filters[1], nb_row=nb_conv[2], nb_col=nb_conv[2], nb_depth=nb_conv[2], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool[2], nb_pool[2], nb_pool[2])))
model.add(Flatten())
model.add(Dense(32, 16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16, nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = RMSprop(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
score, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy = True)
print('Test Accuracy:', accuracy)
