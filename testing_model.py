import cv2  # OpenCV for better preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    img = img.astype("float32") / 255.0

    img_array = np.expand_dims(img, axis=0)

    predictions = model.predict(img_array)
    for i, prob in enumerate(predictions[0]):
        print(f"Digit {i}: {prob:.4f}")

    predicted_label = np.argmax(predictions)

    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted Number: {predicted_label}")
    plt.axis("off")
    plt.savefig("prediction.png")

    return predicted_label

model = tf.keras.models.load_model("mnist_model.keras")
predict_image("image.png", model)