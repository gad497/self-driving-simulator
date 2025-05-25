import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail

def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data.iloc[0], indexed_data.iloc[1], indexed_data.iloc[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data.iloc[3]))
        # image_path.append(os.path.join(datadir, left.strip()))
        # steering.append(float(indexed_data.iloc[3])+0.15)
        # image_path.append(os.path.join(datadir, right.strip()))
        # steering.append(float(indexed_data.iloc[3])-0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

def zoom(image):
    h, w = image.shape[:2]
    scale = 1 - random.random() * 0.3 # Zoom by 0 to +30%
    new_w = int(w*scale)
    new_h = int(h*scale)
    x1 = (w-new_w)//2
    y1 = (h-new_h)//2
    x2 = x1 + new_w
    y2 = y1 + new_h
    return image[y1:y2, x1:x2]

def pan(image):
    h, w = image.shape[:2]
    w_scale = random.random()*0.2 - 0.1
    h_scale = random.random()*0.2 - 0.1
    out = np.zeros_like(image)
    if w_scale >= 0 and h_scale >= 0:
        pan_x = int(w*w_scale)
        pan_y = int(h*h_scale)
        out[pan_y:, pan_x:] += image[:h-pan_y, :w-pan_x]
        return out
    elif w_scale < 0 and h_scale < 0:
        pan_x = int(w*-w_scale)
        pan_y = int(h*-h_scale)
        out[:h-pan_y, :w-pan_x] += image[pan_y:, pan_x:]
        return out
    elif w_scale >=0 and h_scale < 0:
        pan_x = int(w*w_scale)
        pan_y = int(h*-h_scale)
        out[:h-pan_y, pan_x:] += image[pan_y:, :w-pan_x]
        return out
    elif w_scale < 0 and h_scale >=0:
        pan_x = int(w*-w_scale)
        pan_y = int(h*h_scale)
        out[pan_y:, :w-pan_x] += image[:h-pan_y, pan_x:]
        return out
    
def change_brightness(image):
    offset = int(random.random()*0.2*255)
    image_copy = image.copy()
    image_copy[:,:,0] = np.clip(image[:,:,0] - offset, 0, 255)
    return image_copy

def flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = - steering_angle
    return image, steering_angle

def random_augment(image, steering_angle):
    image = cv2.imread(image, cv2.IMREAD_COLOR_RGB)
    if random.random() < 0.5:
        image = change_brightness(image)
    if random.random() < 0.5:
        image = pan(image)
    if random.random() < 0.5:
        image = zoom(image)
    if random.random() < 0.5:
        image, steering_angle = flip(image, steering_angle)
    return image, steering_angle

def image_preprocess(image):
    image = image[60:135, :, :] # remove unnecessary features
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) # nvidia model works best on YUV color space
    image = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.resize(image, (200,66))
    image = image/255
    return image

def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            idx = random.randint(0, len(image_paths)-1)
            if istraining:
                image, steering = random_augment(image_paths[idx], steering_ang[idx])
            else:
                image = cv2.imread(image_paths[idx], cv2.IMREAD_COLOR_RGB)
                steering = steering_ang[idx]
            image = image_preprocess(image)
            batch_img.append(image)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))

# function to create nvidia model
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, 5, strides=(2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, 5, strides=(2,2), activation='elu'))
    model.add(Conv2D(48, 5, strides=(2,2), activation='elu'))
    model.add(Conv2D(64, 3, activation='elu'))
    model.add(Conv2D(64, 3, activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='elu'))
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        loss='mse',
        optimizer=optimizer
    )
    return model

# Read data from csv file
datadir = 'Data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
# pd.set_option('display.max_colwidth', None)
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
# print(data.head())
# Plot the steering angle data
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])),(samples_per_bin, samples_per_bin))
# plt.show()
# Remove data randomly from each bin such that max data is {samples_per_bin}
print('total data:', len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
    random.shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)
print('removed:',len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))
# Plot steering angle data after removing data
hist, bins = np.histogram(data['steering'], num_bins)
plt.figure()
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])),(samples_per_bin, samples_per_bin))
# plt.show()
# Get image_paths and its steering angle from data (X, y)
image_paths, steerings = load_img_steering(os.path.join(datadir,'IMG'), data)
# print('image_paths:',image_paths)
# print('steerings:', steerings)
# Split data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2)
print(f'Training samples: {len(X_train)}, Valid samples: {len(X_val)}')
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
ax[0].set_title('Training set')
ax[1].hist(y_val, bins=num_bins, width=0.05, color='red')
ax[1].set_title('Validation set')
# plt.show()

# Create nvidia model
model = nvidia_model()
print(model.summary())
checkpoint = ModelCheckpoint(
    'best_model.h5',          # Filepath to save the model
    monitor='val_loss',       # Metric to monitor
    save_best_only=True,      # Save only when val_loss improves
    mode='min',               # 'min' because we want lowest loss
    verbose=1
)
history = model.fit(
    batch_generator(X_train, y_train, 128, 1),
    steps_per_epoch=128*5,
    epochs=20,
    validation_data=batch_generator(X_val, y_val, 64, 0),
    validation_steps=64*2,
    verbose=1,
    shuffle=1,
    callbacks=[checkpoint]
)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'val'])
plt.title('Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()
# # model.save('model.h5')