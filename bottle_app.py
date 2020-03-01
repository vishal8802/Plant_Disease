from bottle import route, run, template, response, request, get, post, hook, error
import os
from json import dumps
import numpy as np
import cv2
import io
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from keras import models
import pickle as pkl


# loading model


def load_model():
    global model
    model = models.load_model('./cnn_model.h5')


# image preprocessing
def toArray(filepath):
    try:
        dimension = 256
        image_arr = cv2.imread(filepath)
        image_arr = cv2.resize(image_arr, (dimension, dimension))
        image_arr = np.array(image_arr, dtype=np.float16) / 225.0
        return image_arr.reshape(-1, dimension, dimension, 3)
    except Exception as e:
        print(e)


# class binarization


def binarize():
    label_list = np.load('./GeneratedData/label_as_array.npy')
    label_binarizer = LabelBinarizer()
    label_binarizer.fit_transform(label_list)
    return label_binarizer.classes_


# printing result
def result(p):
    probability = {}
    pred = {}
    label = binarize()
    if np.amax(p) == 0.0:
        print('Unable to predict, All probability are 0')
    else:
        ind = np.where(p == np.amax(p))
        for i in ind[1]:
            #print(f'{label[i]}  ( Accuracy : {(p[0][i]*100):.2f} )')
            pred["prediction"] = f'{label[i]}'
            pred["accuracy"] = f'{(p[0][i]*100):.2f}'
    #print(" ")
    # print("=============\n")
    i = 0
    for l in label:
        probability[f'{l}'] = f'{p[0][i]:.6f}'
        #print(f'{l} ==> {p[0][i]:.6f}\n')
        i = i+1
    return probability, pred


@error(500)
def error_handler_500(error):
    return json.dumps({"success": False, "error": str(error.exception)})


@get('/hello/<name>')
def index(name):
    return template('<b>Hello {{name}}</b>!', name=name)


@post('/predict')
def predict():
    data = {"success": False}
    # reading image
    img = request.files.get('image')

    img.save('./temp/', overwrite=True)

    # preprocessing image
    image = toArray(f'./temp/{img.filename}')
    # print(image.shape)
    # print(image.dtype)
    # print(image)

    res = model.predict(image)
    res, prediction = result(res)
    data["success"] = True
    data["probability"] = res
    data["prediction"] = prediction
    os.remove(f'./temp/{img.filename}')  # delete file function

    response.content_type = 'application/json'
    return dumps(data)


if __name__ == '__main__':
    load_model()
    if os.environ.get('APP_LOCATION') == 'heroku':
        run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    else:
        run(host='localhost', port=8080, debug=True)
