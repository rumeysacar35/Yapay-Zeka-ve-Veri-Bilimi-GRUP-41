from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Girdi modeli
class CareerInput(BaseModel):
    prompt: str

# Model tanımı
career_assistant = pipeline(
    "text2text-generation",
    model="MBZUAI/LaMini-Flan-T5-248M",
    tokenizer="MBZUAI/LaMini-Flan-T5-248M"
)

@app.post("/career")
def get_career_plan(input: CareerInput):
    prompt = (
        f"You are a helpful and concise career assistant. "
        f"Generate a step-by-step career roadmap based on the user's interest. "
        f"Include relevant skills, tools, and resources. Avoid repetition.\n\n"
        f"User input: {input.prompt}"
    )
    
    # Sampling yerine deterministic decoding (top_k=50 + temperature düşürülür)
    result = career_assistant(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.3
    )[0]["generated_text"]
    
    return {"result": result}
