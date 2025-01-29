from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from detect import handle_image_prediction

app = FastAPI()




# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Read the uploaded file
    image_data = await file.read()
    
    # Call the detect.py to process the image
    predicted_class = handle_image_prediction(image_data)
    
    return JSONResponse(content={"message": f"Predicted class: {predicted_class}"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)