import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import re

# set environment variables off that may tamper with computation
os.environ["TF_ENABLE_ONEDNN_OPTS"] ="0" #this line doesnt work lol!

# Define paths
data_dir = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/ut-zap50k-images" #change path as needed
imagepath_mat_file = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/image-path.mat"  # Path to the imagepath.mat file, change as needed

# Load image paths and labels from imagepath.mat file
imagepath_data = loadmat(imagepath_mat_file)
file_paths = imagepath_data['imagepath'].flatten()

#convert to string
file_paths = np.array([str(path[0]) for path in file_paths])

# Extracting the labels from the file paths using regular expressions
labels = [re.match(r'^([^/]+)/', filepath).group(1) for filepath in file_paths] #regEx line to pull the correct term from the file path
# Assuming you have already loaded or generated the labels array coding troubleshoot, nothing to see here

# Print the first five labels
# print("First five labels:")
# for label in labels[:5]:
#     print(label)

# Create a DataFrame from labels and file paths
labels_df = pd.DataFrame({'label': labels, 'filepath': file_paths})

# Assuming the first column contains image filenames and the second column contains labels
label_to_int = {label: idx for idx, label in enumerate(labels_df['label'].unique())}

# Preprocess the data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=data_dir,
    x_col='filepath',  # Column containing image filepaths
    y_col='label',     # Column containing labels
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,      # Shuffle the training data
    classes=list(label_to_int.keys())
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=data_dir,
    x_col='filepath',  # Column containing image filepaths
    y_col='label',     # Column containing labels
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True,      # Shuffle the validation data
    classes=list(label_to_int.keys())
)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(labels_df['filepath'], labels_df['label'], test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(label_to_int), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model
model.save('shoe_classifier_model.h5')
