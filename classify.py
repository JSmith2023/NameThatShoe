import numpy as np
from tensorflow import keras
from keras import models
from keras.models import load_model
from keras.preprocessing import image

def predict_image(image_path):
    #Define a structure for labels(int)
    label_to_int = {'Shoes': 0, 'Boots': 1, 'Slippers': 2, 'Sandals': 3}
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    #img_array = preprocess_input(img_array)
    
    #Reshape
    img_array = img_array.reshape((1,) + img_array.shape)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Process predictions
    predicted_class_label = list(label_to_int.keys())[np.argmax(predictions)]
    print("Predicted class:", predicted_class_label)
    return predicted_class_label, predictions

model_path = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/shoe_classifier_model.h5"
# Load the model
model = models.load_model(model_path)

# Compile the model with the appropriate optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

image_path = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/boot1_iso.png"
predicted_class_label, predictions = predict_image( image_path)

image_path = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/sandal1_iso.png"
predicted_class_label, predictions = predict_image( image_path)

image_path = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/sandal2_iso.png"
predicted_class_label, predictions = predict_image( image_path)

image_path = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/slipper1_iso.png"
predicted_class_label, predictions = predict_image( image_path)

image_path = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/sneaker1_iso.png"
predicted_class_label, predictions = predict_image( image_path)

image_path = "C:/Users/Jash/Documents/HW/NameThatShoe/NameThatShoe/sneaker2_iso.png"
predicted_class_label, predictions = predict_image( image_path)


