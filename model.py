import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers.convolutional import Convolution2D
from jinja2.nodes import Output


log_path = 'data/driving_log.csv'
log_path2 = 'data2/driving_log.csv'
log_path3 = 'data3/driving_log.csv'
log_path4 = 'data4/driving_log.csv'

def readData(log_path):
    output_lines = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            output_lines.append(line)
    return output_lines

def measurementCorrection(data, name, corfactor):
    if name==0:
        return float(data[3])
    if name==1:
        return float(data[3]) + corfactor
    if name==2:
        return float(data[3]) - corfactor

lines = readData(log_path) + readData(log_path2) + readData(log_path3) + readData(log_path4)
images = []
measurements = []
for line in lines:
    # Center/Left/Right camera image
    for index in range(3):
        source_path = line[index]
        filename = source_path.split('/')[-1]
        image = cv2.imread(filename)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image2)
        # Different camera position need different steering angle
        # 0.2 is an estimation, not calculated!
        measurement = measurementCorrection(line, index, 0.2)
        measurements.append(measurement)

# Flipping all image and steering measurment along y axis at center
augmented_images, augmented_measurements = [], []
for image in images:
    augmented_images.append(image)
    augmented_images.append(np.fliplr(image))
    
for measurement in measurements:
    augmented_measurements.append(measurement)
    augmented_measurements.append(-measurement)

# Final training data
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Dropout, Flatten, Dense

# Model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24,(5,5),activation="relu", strides=(2,2)))
model.add(Convolution2D(36,(5,5),activation="relu", strides=(2,2)))
model.add(Convolution2D(48,(5,5),activation="relu", strides=(2,2))) 
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50)) 
model.add(Dense(1)) 

# Training
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, epochs=5, verbose=1, validation_split=0.2, shuffle=True)

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save model
model.save('model.h5')
print("Model saved")
