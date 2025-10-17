from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms
from PIL import Image
import torch
import spacy

from app.bigram_model import BigramModel
from app.cnn_model import SimpleCNN

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

nlp = spacy.load("en_core_web_lg")
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int
class EmbeddingRequest(BaseModel):
    text: str
class EmbeddingResponse(BaseModel):
    vector: List[float]
    
cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load("app/cnn_cifar10.pth", map_location=torch.device("cpu")))
cnn_model.eval()
transforms_pipeline = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
    
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.text)
    return {"vector": doc.vector.tolist()}

@app.post("/predict_cnn")
async def predict_cnn(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transforms_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CIFAR10_CLASSES[predicted.item()]
    return {"prediction": predicted_class}