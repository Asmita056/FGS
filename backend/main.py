# FASTAPI

import numpy as np
import aiofiles
import pickle
import os
import random
# from PIL import image
from fastapi import FastAPI,File,Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from train_model import predict_image
# from io import BytesIO


# print("helooooooooooo", tensorflow.__version__)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

UPLOAD_FOLDER = 'C:/Users/akash/OneDrive/Desktop/Fruit Grading System/FGS/uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Preload your models from the pickle file
with open('fruit_grading_system_models.pkl', 'rb') as f:
    models = pickle.load(f)
    cnn_model = models['cnn_model']
    knn = models['knn']
    nb = models['nb']
    rf = models['rf']
    svm = models['svm']
    log_reg_model = models['log_reg_model']
    ensemble = models['ensemble']
    grad_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)  # Feature extractor
    label_encoder = models['label_encoder']

categories = ['Best', 'Average', 'Worst']

bestTotal = 0
averageTotal = 0
worstTotal = 0
totalCount = 0

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...), 
    username: str = Form(...)):

    global bestTotal, averageTotal, worstTotal, totalCount

    random_number = random.randint(100000, 999999)
    file_extension = os.path.splitext(file.filename)[1]
    new_filename = f"{random_number}{file_extension}"
    file_location = f"{UPLOAD_FOLDER}/{new_filename}"
    
    async with aiofiles.open(file_location, 'wb') as f:
        content = await file.read() 
        await f.write(content) 

    # Preprocess the image and make prediction
    img = load_img(file_location, target_size=(128, 128))  
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    # Call your prediction function
    # predictions = predict_image(img_array)
    predicted_category = predict_image(img_array)

    totalCount += 1
    if predicted_category == 'Best':
        bestTotal += 1
    elif predicted_category == 'Average':
        averageTotal += 1
    elif predicted_category == 'Worst':
        worstTotal += 1

    
    return {"message": f"File uploaded successfully by {username}", 
    "predictions" : predicted_category,
    "total_count": totalCount,
    "best_count": bestTotal,
    "average_count": averageTotal,
    "worst_count": worstTotal,
    "file_path": file_location, 
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)