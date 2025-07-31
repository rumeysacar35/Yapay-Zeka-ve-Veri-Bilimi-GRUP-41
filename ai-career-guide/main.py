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
    lang:str

# Model tanımı
career_assistant = pipeline(
    "text2text-generation",
    model="MBZUAI/LaMini-Flan-T5-248M",
    tokenizer="MBZUAI/LaMini-Flan-T5-248M",
    device=-1
)


# Türkçe → İngilizce çeviri
tr_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-tr-en", device=-1)

# İngilizce → Türkçe çeviri
en_tr_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr",device=-1)


@app.post("/career")
async def generate_career_path(input_data: CareerInput):
    # input_data.prompt doğrudan frontend’den geliyor
    # "Türkçe cevap ver. ..." veya sadece İngilizce prompt olabilir

    # Burada dil kontrolü yapmaya gerek yok, çünkü ön-ek frontend’den geliyor.

    # Önce Türkçe ise İngilizceye çeviri yap, yoksa direkt devam et
    if input_data.prompt.startswith("Türkçe cevap ver."):
        # Prefix kaldır
        clean_prompt = input_data.prompt.replace("Türkçe cevap ver. ", "")
        tr_en_translation = tr_en_translator(clean_prompt)[0]["translation_text"]
    else:
        tr_en_translation = input_data.prompt

    # Modelle devam et
    result = career_assistant(
        f"You are a helpful and concise career assistant. Generate a step-by-step career roadmap based on the user's interest. Include relevant skills, tools, and resources. Avoid repetition.\n\nUser input: {tr_en_translation}",
        max_new_tokens=200,
        do_sample=True,
        top_k=50,
        temperature=0.4,
        repetition_penalty=1.8
    )[0]["generated_text"]

    # İngilizce → Türkçe çeviri sadece Türkçe isteği için
    if input_data.prompt.startswith("Türkçe cevap ver."):
        translated = en_tr_translator(result)[0]["translation_text"]
        return {"result": translated}
    else:
        return {"result": result}