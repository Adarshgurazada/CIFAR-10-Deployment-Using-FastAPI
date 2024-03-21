from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from functions import preprocess_image
import numpy as np
import cv2  

# app = FastAPI()
# model = load_model('vgg19_model.h5')

# @app.post("/predict/")
# async def predict_image(file: UploadFile = File(...)):
#     image = await file.read()
#     # Decode and preprocess the uploaded image
#     decoded_image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
#     processed_image = preprocess_image(decoded_image)
#     # Make predictions using the loaded model
#     prediction = model.predict(np.expand_dims(processed_image, axis=0))
#     predicted_class = np.argmax(prediction)
#     return {"predicted_class": int(predicted_class)}


from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from functions import preprocess_image
import numpy as np

app = FastAPI()
model = load_model('vgg19_model.h5')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = await file.read()
    # Decode and preprocess the uploaded image
    decoded_image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    processed_image = preprocess_image(decoded_image)
    # Make predictions using the loaded model
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class_idx = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_idx]
    return {"predicted_class": predicted_class_name}
