import csv
import numpy as np
import sklearn

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dropout, Dense, Cropping2D
from keras.layers import Lambda


MAIN_DIR = "/workspace/data/"
BATCH_SIZE = 16


def load_lines(dir_loc=MAIN_DIR):
    lines = []
    with open(dir_loc + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for count, line in enumerate(reader):
            if count > 0:
                lines.append(line)
            else:
                print(line)
    return lines


def translate_image(image, angle, translation_range=10):
    rows, cols = image[:, :, 1].shape
    tranlation_x = (translation_range * np.random.uniform()) - (translation_range / 2)
    angle_adjusted = angle + (tranlation_x / translation_range * 2 * .2)
    translation_y = (40 * np.random.uniform()) - (40 / 2)
    translation_matrix = np.float32([[1, 0, tranlation_x], [0, 1, translation_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

    return translated_image, angle_adjusted


def flip_image(image, angle):
    image = cv2.flip(image, 1)  # or np.fliplr(image)
    angle = angle * -1.0
    return image, angle


def generator(samples, batch_size=64, correction=0.25, dir_loc=MAIN_DIR):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:(offset + batch_size)]

            images = []
            angles = []

            draws = np.random.uniform(size=len(batch_samples)) - 0.5
            for i, batch_sample in enumerate(batch_samples):
                center_image = mpimg.imread(dir_loc + batch_sample[0].strip())
                left_image = mpimg.imread(dir_loc + batch_sample[1].strip())
                right_image = mpimg.imread(dir_loc + batch_sample[2].strip())

                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                center_image, center_angle = translate_image(center_image, center_angle)
                left_image, left_angle = translate_image(left_image, left_angle)
                right_image, right_angle = translate_image(right_image, right_angle)

                if(draws[i] > 0):
                        center_image, center_angle = flip_image(center_image, center_angle)
                        left_image, left_angle = flip_image(left_image, left_angle)
                        right_image, right_angle = flip_image(right_image, right_angle)

                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def load_generator(dir_loc=MAIN_DIR, test_size=0.2):
    samples = load_lines(dir_loc)
    train_samples, validation_samples = train_test_split(samples, test_size=test_size)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
    return train_generator, validation_generator, len(train_samples), len(validation_samples)


def steering_model(visualize=False):
    input_shape = (160, 320, 3)
    kernel_size = (3, 3)
    depth = (64, 32, 16, 8)
    pool_size = (2, 2)

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Conv2D(depth[0], kernel_size, padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(depth[1], kernel_size))
    model.add(Activation('relu'))
    model.add(Conv2D(depth[2], kernel_size))
    model.add(Activation('relu'))
    model.add(Conv2D(depth[3], kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    if visualize:
        model.summary()

    return model


def fit_model(output_name="model.h5", dir_loc=MAIN_DIR):
    train_generator, validation_generator, len_train, len_validation = load_generator()
    model = steering_model()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=2 * len_train // 16,
                                  validation_data=validation_generator,
                                  validation_steps=2 * len_validation // 16,
                                  epochs=5)
    model.save(dir_loc + output_name)
    return history


def visualize_model(model_history):
    # plot the training and validation loss for each epoch
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
