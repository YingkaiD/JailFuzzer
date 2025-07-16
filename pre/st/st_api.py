import torch
import json
import uvicorn
from pydantic import BaseModel
from fastapi import Response
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelWorker():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def generate_cosine_similarity(self, sentence1: str, sentence2: str):
        # embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        embeddings_1 = self.model.encode(sentence1, convert_to_tensor=True).to(device)
        embeddings_2 = self.model.encode(sentence2, convert_to_tensor=True).to(device)

        cosine_scores = util.pytorch_cos_sim(embeddings_1, embeddings_2)
        return cosine_scores.tolist()

class ListPrompt(BaseModel):
    content_prompt_1: list
    content_prompt_2: list

@app.post("/generate_cosine_similarity")
async def generate_cosine_similarity(sentence: ListPrompt):
    sentence_1 = sentence.content_prompt_1
    sentence_2 = sentence.content_prompt_2
    cosin_scores = worker.generate_cosine_similarity(sentence1=sentence_1, sentence2=sentence_2)
    json_content = json.dumps(cosin_scores)
    return Response(content=json_content, media_type="application/json")

if __name__ == '__main__':

    worker = ModelWorker()

    uvicorn.run(app, host="0.0.0.0", port=22001, log_level="info")