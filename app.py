import os

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from deepface.basemodels import Facenet
from deepface.commons import functions, realtime, distance as dst
from deepface import DeepFace
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from deepface.basemodels import Facenet
import numpy as np
import pickle
app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
# cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "facenet.h5")
loaded_model = pickle.load(open(STATIC_FOLDER + "/models/" + "facenet.h5", 'rb'))
# loaded_model = pickle.load(open(STATIC_FOLDER + "/models/" + "vggface.h5", 'rb'))
IMAGE_SIZE = 64

# Preprocess an image
# def preprocess_image(image):
#     # image = tf.image.decode_jpeg(image, channels=1)
#     # image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#     # image /= 255.0  # normalize to [0,1] range
    

#     return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    # image = tf.io.read_file(path)
    embedding = DeepFace.represent(img_path = path , model_name = 'Facenet')
    samples = expand_dims(embedding, axis=0)
    return samples


# Predict & classify image
def classify(model, image_path):
    label = 0
    classified_pred = 0

    preprocessed_imgage = load_and_preprocess_image(image_path)
    
    # preprocessed_imgage = tf.reshape(
    #     preprocessed_imgage, (1,IMAGE_SIZE, IMAGE_SIZE)
    # )
    labels = ["Ben Afflek", "Gautam Rode", "Jerry Seinfeld", "Madona", "Mindy Kaling", "The Rock", "Thomas Shelby"]
    yhat_class = loaded_model.predict(preprocessed_imgage)
    yhat_prob = loaded_model.predict_proba(preprocessed_imgage)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predicted_label = labels[class_index]
    
    
    
    
        
    return predicted_label, class_probability


# home page
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        upload_image_path .encode('utf-8')
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(loaded_model, upload_image_path)

        prob = np.round(prob, 2)
        

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True