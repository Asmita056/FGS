from fastapi import FastAPI,File,Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# from PIL import image
import aiofiles
import os
import random

app = FastAPI()

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
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

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...), 
    username: str = Form(...)):
    random_number = random.randint(100000, 999999)
    file_extension = os.path.splitext(file.filename)[1]
    new_filename = f"{random_number}{file_extension}"
    file_location = f"{UPLOAD_FOLDER}/{new_filename}"
    
    async with aiofiles.open(file_location, 'wb') as f:
        content = await file.read() 
        await f.write(content) 
    
    return {"message": f"File uploaded successfully by {username}", "file_path": file_location}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

