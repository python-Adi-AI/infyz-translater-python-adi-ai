import time
import hashlib
import torch
import redis
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable CUDA (GPU) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the translation model and tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Attempt model compilation (if supported)
try:
    model = torch.compile(model)
except Exception as e:
    print(f"⚠️ Model compilation failed: {e}. Running without compilation.")

# Enable TensorFloat32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True

# Initialize Redis connection
# try:
#     redis_cache = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
#     if redis_cache.ping():
#         print("✅ Redis connected successfully!")
# except Exception as e:
#     print(f"⚠️ Redis connection failed: {e}")
#     redis_cache = None  # Disable caching if Redis is not available

# Initialize Redis connection with password
try:
    redis_cache = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        password="aiteam-redis",  # Add your Redis password here
        decode_responses=True
    )
    if redis_cache.ping():
        print("✅ Redis connected successfully!")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    redis_cache = None  # Disable caching if Redis is not available



# Supported language codes
LANG_CODE_MAP = {
    'en': 'en', 'hi': 'hi', 'pl': 'pl', 'nl': 'nl', 'es': 'es'
}

# Define Input Schema
class TranslationRequest(BaseModel):
    text: List[str]
    source_lang: str
    target_lang: str

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v:
            raise ValueError('Text list cannot be empty')
        if not all(isinstance(item, str) for item in v):
            raise ValueError('All text items must be strings')
        return v

    @field_validator('source_lang', 'target_lang')
    @classmethod
    def validate_language(cls, v):
        if v not in LANG_CODE_MAP:
            raise ValueError(f'Unsupported language code. Supported: {list(LANG_CODE_MAP.keys())}')
        return v

# Redis Caching Decorator
def cached_translation(func):
    def wrapper(texts, source_lang, target_lang):
        if redis_cache is None:
            return func(texts, source_lang, target_lang)

        cache_keys = [f"trans:{source_lang}:{target_lang}:{hashlib.md5(t.encode()).hexdigest()}" for t in texts]
        cached_results, uncached_texts, uncached_indices = [], [], []
        print("=-----------------cached_results ==============",cached_results)

        # Check cache for each key
        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            cached_value = redis_cache.get(key)
            if cached_value:
                cached_results.append(cached_value)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Fetch translations for uncached texts
        if uncached_texts:
            new_translations = func(uncached_texts, source_lang, target_lang)
            for i, (key, translation) in enumerate(zip([cache_keys[i] for i in uncached_indices], new_translations)):
                redis_cache.setex(key, 2592000, translation)  # Cache for 1 hour  -? 30 days chainge 
                cached_results.insert(uncached_indices[i], translation)

        return cached_results
    return wrapper

# Text Translation Function
@cached_translation
def translate_text(texts, source_lang, target_lang):
    try:
        tokenizer.src_lang = LANG_CODE_MAP[source_lang]
        target_lang_code = tokenizer.lang_code_to_id[LANG_CODE_MAP[target_lang]]

        def chunk_texts(texts, max_tokens=None, overlap=50):
            chunked_texts = []
            for text in texts:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                # If no max_tokens specified, use the entire text
                if max_tokens is None or len(tokens) <= max_tokens:
                    chunked_texts.append(text)
                else:
                    # Adaptive chunking with dynamic chunk size
                    chunks = []
                    for i in range(0, len(tokens), max_tokens - overlap):
                        chunk_tokens = tokens[i:i+max_tokens]
                        chunks.append(tokenizer.decode(chunk_tokens))
                    chunked_texts.extend(chunks)
            
            return chunked_texts

        # Dynamically determine max tokens based on model's max length
        max_model_tokens = model.config.max_position_embeddings
        chunked_texts = chunk_texts(texts, max_tokens=max_model_tokens)

        translations = []
        for chunk in chunked_texts:
            try:
                inputs = tokenizer([chunk], return_tensors="pt", padding=True, truncation=False).to(device)
                generated_tokens = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    forced_bos_token_id=target_lang_code,
                    num_beams=4,
                    early_stopping=True,
                    use_cache=True,
                    do_sample=False,
                    max_length=max_model_tokens,
                    no_repeat_ngram_size=2
                )
                translations.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])
            except Exception as chunk_error:
                print(f"⚠️ Chunk Translation Error: {chunk_error}")
                translations.append(chunk)  # Fallback to original chunk

        # Intelligent chunk merging
        final_translation = " ".join(translations)
        return [final_translation]
    except Exception as e:
        print(f"⚠️ Translation Error: {str(e)}")
        return [f"⚠️ Translation Error: {str(e)}" for _ in texts]

# API Endpoint
@app.post("/translate")
async def translation_endpoint(request: TranslationRequest):
    try:
        start_time = time.time()
        print(f"\n🔹 Processing {len(request.text)} texts from {request.source_lang} → {request.target_lang}")

        translations = translate_text(request.text, request.source_lang, request.target_lang)

        elapsed_time = time.time() - start_time
        print(f"✅ Completed in {elapsed_time:.4f} seconds.")

        return {
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "translations": translations,
            "processing_time": f"{elapsed_time:.4f} seconds",
            "text_count": len(request.text)
        }
    except Exception as e:
        print(f"❌ Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI Server
if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("server.log")]
    )

    try:
        print("🚀 Starting FastAPI Server...")
        uvicorn.run("app:app", host="0.0.0.0", port=5000, workers=4, reload=True)
    except Exception as e:
        logging.error(f"❌ Server Startup Failed: {e}", exc_info=True)
        print(f"❌ Critical Server Startup Error: {e}")
        sys.exit(1)
