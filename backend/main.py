from fastapi import FastAPI,File,Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import image
import io
app = FastAPI()

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
    "http://localhost:3000",
]


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


@app.post("/uploadtwo")
async def recievetwo(file: bytes=File(...),username :str=Form(...)):
     print(username)
     image= image.open(io.BytesIO(file))
     image.show()

     return {"uploadStatus":"Complete"}
     print(file)


@app.post("/upload")
async def recieverfile(file: bytes=File(...)):
     image= image.open(io.BytesIO(file))
     image.show()
     return{"uploadStatus":"complete"}
     print(file)

