from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from ultralytics import YOLO
import uuid

app = FastAPI()

# CORS agar bisa diakses dari luar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("my_model.pt")

label_map = ["100k", "10k", "1k", "20k", "2k", "50k", "5k"]
label_dict = {
    "100k": "seratus ribu",
    "10k": "sepuluh ribu",
    "1k": "seribu",
    "20k": "dua puluh ribu",
    "2k": "dua ribu",
    "50k": "lima puluh ribu",
    "5k": "lima ribu"
}

# Ganti suara dengan log teks
def speak(text):
    print(f"[Simulasi suara]: {text}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_path)[0]
    os.remove(temp_path)

    if results.boxes:
        class_id = int(results.boxes.cls[0])
        label_raw = label_map[class_id]
        label_speak = label_dict[label_raw]
        speak(f"{label_speak} rupiah")
        return {"detected": label_raw}
    else:
        return {"detected": "Tidak terdeteksi"}
