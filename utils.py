import tensorflow as tf
import json
import sys
import numpy as np
import argparse
import pickle
from PIL import Image
import cv2

with open('age_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

gender_labels = {
    1: 'Male',
    0: 'Female'
}

gender_model = tf.keras.models.load_model('cnn_simple_gender.h5')
age_model = tf.keras.models.load_model('age_prediction_agu_200.h5')


def preprocess_image(img_):
    img = tf.image.resize(img_, [128, 128])
    img = tf.expand_dims(img, axis=0)
    return img


def predict_gender(roi_color):
    gender_prediction = gender_model.predict(preprocess_image(roi_color))
    gender_prediction = gender_labels[np.argmax(gender_prediction)]
    return gender_prediction


def predict_age(roi_color):
    age_prediction = age_model.predict(preprocess_image(roi_color))
    age_prediction = le.inverse_transform([np.argmax(age_prediction)])
    return age_prediction
