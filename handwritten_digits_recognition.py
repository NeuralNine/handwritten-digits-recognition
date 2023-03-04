import os
import cv2
import numpy as np
# Turning of the TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import matplotlib.pyplot as plt

print("-" * 70)
print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")
print("-" * 70 + "\n")

# Decide if to load an existing model or to train a new one
while True:
    train_new_model = input("[*] Do you want to train a new model (You can use existing one) [Y]/[N]? ")
    
    if train_new_model.upper() == "Y":
        train_new_model = True
        print("[I] The new model is being trained! \n")
        break
    elif train_new_model.upper() == "N":
        print("[I] The existing model is being used! \n")
        train_new_model = False 
        break
    else:
        print("\n")
        print("[!] Please just type Y or N! \n")

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print("="*50)
    print(f"[*] Validation Loss: {val_loss}")
    print("-"*50)
    print(f"[*] Vaidation Accuracy: {val_acc}")
    print("="*50 + "\n")

    # Saving the model
    model.save('../handwritten-digits/handwritten_digits.model', overwrite=True)
else:
    # Load the model
    model = tf.keras.models.load_model('../handwritten-digits/handwritten_digits.model')

# Load custom images and predict them
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("[*] The number is probably a {} \n".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("[!] Error reading image! Proceeding with next image... \n")
        image_number += 1
