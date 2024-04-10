import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the data
data_dir = '/path/to/your/dataset'
image_paths = []
labels = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))
            labels.append(os.path.basename(root))

# Convert labels to integers
label_to_int = {label: idx for idx, label in enumerate(np.unique(labels))}
int_to_label = {idx: label for label, idx in label_to_int.items()}
labels = [label_to_int[label] for label in labels]

# Load and preprocess images using OpenCV
images = []
for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Resize images to 150x150
    img = img.astype('float32') / 255.0  # Normalize pixel values
    images.append(img)

images = np.array(images)
labels = np.array(labels)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

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
    Dense(50, activation='softmax')  # Assuming 50 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_val, y_val)
)

# Save the model
model.save('shoe_classifier_model.h5')
