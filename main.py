import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    model = load_model('model.h5')
    tensor1 = model.predict(face_list1)
    tensor2 = model.predict(face_list2)

    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def update_item(file1: UploadFile, file2: UploadFile):
    content1 = await file1.read()
    nparr1 = np.frombuffer(content1, np.uint8)
    image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

    content2 = await file2.read()
    nparr2 = np.frombuffer(content2, np.uint8)
    image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # image1 = cv2.imread("img1.png")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image1, (128, 128))
    image1 = np.expand_dims(image1, axis=0)
    # image2 = cv2.imread("img2.png")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image2, (128, 128))
    image2 = np.expand_dims(image2, axis=0)
    print(image1.shape)
    print(image2.shape)
    result = classify_images(image1, image2)
    print(result)
    return {"verdict": True if result[0] else False}
