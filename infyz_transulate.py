import os
import sys
import io
import json
import torch
import logging
import logging.handlers
import traceback
import redis
import hashlib
import threading
import time
import atexit
import shutil
from datetime import datetime, timedelta
from functools import lru_cache
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import functools
import asyncio
import concurrent.futures
from typing import List, Dict, Any

# Configure logging with multiple handlers and detailed formatting
def cleanup_old_logs(log_dir, max_age_hours=24):
    """
    Remove log files older than specified hours.
    
    :param log_dir: Directory containing log files
    :param max_age_hours: Maximum age of log files in hours
    """
    try:
        current_time = datetime.now()
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
            
            # Get file modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Check if file is older than max_age_hours
            if current_time - mod_time > timedelta(hours=max_age_hours):
                try:
                    os.remove(file_path)
                    logging.info(f"Removed old log file: {filename}")
                except Exception as e:
                    logging.error(f"Error removing log file {filename}: {e}")
    except Exception as e:
        logging.error(f"Log cleanup failed: {e}")

def start_log_cleanup_thread(log_dir, interval_hours=24):
    """
    Start a background thread to periodically clean up old log files.
    
    :param log_dir: Directory containing log files
    :param interval_hours: Interval between log cleanups
    """
    def cleanup_thread():
        while True:
            cleanup_old_logs(log_dir)
            time.sleep(interval_hours * 3600)  # Convert hours to seconds
    
    # Create and start the daemon thread
    thread = threading.Thread(target=cleanup_thread, daemon=True)
    thread.start()
    return thread

def setup_logging(log_level=logging.INFO):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler for general logs
            logging.FileHandler(os.path.join(log_dir, 'app.log'), encoding='utf-8'),
            # Rotating file handler for error logs
            logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, 'error.log'), 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
        ]
    )

    # Start log cleanup thread
    cleanup_thread = start_log_cleanup_thread(log_dir)
    
    # Ensure logs are cleaned up on exit
    atexit.register(cleanup_old_logs, log_dir)
    
    return cleanup_thread

# Call setup logging when the module is imported
log_cleanup_thread = setup_logging()

# Enhanced error logging function
def log_exception(message, exc_info=True):
    """
    Log an exception with a custom message and optional traceback.
    
    :param message: Custom error message
    :param exc_info: Whether to include exception traceback
    """
    try:
        logging.error(message, exc_info=exc_info)
    except Exception as log_error:
        print(f"Logging failed: {log_error}")
        print(f"Original error message: {message}")

# Set UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure log directory exists
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'translation_debug.log'), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Redis Cache Configuration
try:
    redis_cache = redis.Redis(host='localhost', port=6379, db=0)
except Exception as e:
    logger.warning(f"Redis cache initialization failed: {e}")
    redis_cache = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Comprehensive Language Mapping
languages = {
    'en': {'name': 'English', 'nativeName': 'English', 'flag': '🇺🇸'},
    'fr': {'name': 'French', 'nativeName': 'Français', 'flag': '🇫🇷'},
    'de': {'name': 'German', 'nativeName': 'Deutsch', 'flag': '🇩🇪'},
    'es': {'name': 'Spanish', 'nativeName': 'Español', 'flag': '🇪🇸'},
    'it': {'name': 'Italian', 'nativeName': 'Italiano', 'flag': '🇮🇹'},
    'pt': {'name': 'Portuguese', 'nativeName': 'Português', 'flag': '🇵🇹'},
    'hi': {'name': 'Hindi', 'nativeName': 'हिन्दी', 'flag': '🇮🇳'},

}

# Advanced Caching Decorator
def cached_translation(func):
    @functools.wraps(func)
    def wrapper(text, source_lang, target_lang, *args, **kwargs):
        # Generate a unique cache key
        cache_key = hashlib.md5(
            f"{text}|{source_lang}|{target_lang}".encode()
        ).hexdigest()
        
        # Check Redis cache first
        try:
            cached_result = redis_cache.get(cache_key)
            if cached_result:
                return cached_result.decode('utf-8')
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        # Perform translation
        result = func(text, source_lang, target_lang, *args, **kwargs)
        
        # Cache the result
        try:
            redis_cache.setex(cache_key, 3600, result)  # Cache for 1 hour
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
        
        return result
    return wrapper

# Optimized Batch Translation
def fast_batch_translate(
    texts: List[str], 
    source_lang: str, 
    target_lang: str, 
    batch_size: int = 32
) -> List[str]:
    """
    Perform fast batch translation with minimal overhead
    """
    translations = []
    
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Split texts into batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Prepare batch translation futures
            futures = [
                executor.submit(translate_text, text, source_lang, target_lang)
                for text in batch
            ]
            
            # Collect results
            batch_translations = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
            
            translations.extend(batch_translations)
    
    return translations

# Async Translation Endpoint
@app.route('/async_translate', methods=['POST'])
async def async_translate_endpoint():
    """
    Asynchronous translation endpoint for high-performance translation
    """
    data = request.get_json()
    texts = data.get('texts', [])
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'fr')
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    
    try:
        # Use asyncio for concurrent processing
        loop = asyncio.get_event_loop()
        translations = await loop.run_in_executor(
            None, 
            fast_batch_translate, 
            texts, 
            source_lang, 
            target_lang
        )
        
        return jsonify({
            "translations": translations,
            "source_lang": source_lang,
            "target_lang": target_lang
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optimized Single Translation with Caching
@cached_translation
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Optimized translation function with caching and performance improvements
    """
    try:
        # Set source language
        tokenizer.src_lang = source_lang
        
        # Tokenize input with minimal preprocessing
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=50000
        )
        
        # Generate translation with optimized parameters
        translation_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            max_length=len(inputs.input_ids[0]) + 50,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode translation
        translation = tokenizer.decode(
            translation_ids[0], 
            skip_special_tokens=True
        )
        
        return translation
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

# Language Detection Endpoint
@app.route('/detect_language', methods=['POST'])
def detect_language_endpoint():
    """
    Fast language detection endpoint
    """
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Use a simple language detection approach
        detected_lang = detect_language(text)
        
        return jsonify({
            "detected_language": detected_lang,
            "language_info": languages.get(detected_lang, {})
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def detect_language(text: str) -> str:
    """
    Simple language detection function
    """
    # This is a placeholder. In production, use a more sophisticated library
    # like langdetect or polyglot
    try:
        from langdetect import detect
        return detect(text)
    except ImportError:
        # Fallback to a simple heuristic
        return 'en'  # Default to English if detection fails

# Performance Monitoring Middleware
@app.before_request
def log_request_info():
    """
    Log performance metrics for each request
    """
    request.start_time = time.time()

@app.after_request
def log_response_time(response):
    """
    Log and track response times
    """
    request_time = time.time() - request.start_time
    logger.info(f"Request to {request.path}: {request_time:.4f} seconds")
    return response

# Optimize model loading with caching and compilation
@lru_cache(maxsize=1)
def load_model_and_tokenizer(model_name):
    try:
        logger.info(f"Attempting to load model: {model_name}")
        
        # Determine optimal device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model and tokenizer
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        
        # Move model to device
        model = model.to(device)

        # PyTorch 2.0 Compilation for faster inference
        try:
            model = torch.compile(model)
            logger.info("Model compiled successfully with torch.compile()")
        except Exception as compile_error:
            logger.warning(f"Model compilation failed: {compile_error}")

        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Could not load model: {e}")

# Global model and tokenizer
MODEL_NAME = "facebook/m2m100_418M"
try:
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)
except Exception as e:
    logger.critical(f"Model initialization failed: {e}")
    model, tokenizer, device = None, None, None

# Batch translation endpoint with async support
@app.route('/translate_batch', methods=['POST'])
def translate_batch_texts():
    data = request.get_json(force=True)
    
    texts = data.get('texts', [])
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'en')
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    
    translations = fast_batch_translate(texts, source_lang, target_lang)
    return jsonify({"translations": translations})

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    start_time = time.time()
    try:
        # Enhanced request parsing with detailed logging
        data = request.get_json(force=True)
        
        # Validate input parameters
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'en').lower()
        target_lang = data.get('target_lang', 'en').lower()
        
        # Input validation
        if not text:
            logger.warning(f"Empty translation request: source={source_lang}, target={target_lang}")
            return jsonify({
                "error": "No text provided for translation",
                "status": "failed"
            }), 400
        
        # Detect source language if not specified
        if source_lang == 'auto':
            source_lang = detect_language(text)
        
        # Perform translation
        translation = translate_text(text, source_lang, target_lang)
        
        # Performance tracking
        end_time = time.time()
        translation_time = end_time - start_time
        
        # Detailed logging
        logger.info(f"Translation Request Details:")
        logger.info(f"Source Language: {source_lang}")
        logger.info(f"Target Language: {target_lang}")
        logger.info(f"Text Length: {len(text)} characters")
        logger.info(f"Translation Time: {translation_time:.4f} seconds")
        
        # Enhanced response with metadata
        return jsonify({
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text_length": len(text),
            "translation_time": translation_time,
            "status": "success"
        }), 200
    
    except json.JSONDecodeError:
        logger.error("Invalid JSON in translation request")
        return jsonify({
            "error": "Invalid JSON format",
            "status": "failed"
        }), 400
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Translation failed: {str(e)}",
            "status": "failed"
        }), 500

# Enhanced languages endpoint with POST support
@app.route('/languages', methods=['GET', 'POST'])
def handle_languages():
    if request.method == 'POST':
        # Parse JSON data safely
        try:
            data = request.get_json(force=True, silent=True) or {}
        except Exception as json_error:
            logger.error(f"JSON parse error: {json_error}")
            return jsonify({"error": "Malformed request"}), 400
        
        # Filter languages based on optional criteria
        search_term = data.get('search', '').lower()
        limit = data.get('limit', len(languages))
        
        # Filter and prepare languages
        filtered_languages = [
            {
                'code': code, 
                'name': lang_info['name'], 
                'nativeName': lang_info['nativeName'], 
                'flag': lang_info['flag']
            }
            for code, lang_info in languages.items()
            if (not search_term or 
                search_term in lang_info['name'].lower() or 
                search_term in lang_info['nativeName'].lower())
        ][:limit]
        
        return jsonify(filtered_languages)
    
    # Default GET method behavior
    return jsonify(list(languages.keys()))

# Optimized Translation Processor
def fast_translation_processor(
    text: str, 
    source_lang: str = None, 
    target_lang: str = 'en'
):
    """
    Ultra-fast translation processor with minimal overhead
    """
    try:
        # Quick language detection
        if not source_lang:
            source_lang = detect_language(text)
        
        # Validate and sanitize languages
        source_lang = source_lang if source_lang in languages else 'auto'
        target_lang = target_lang if target_lang in languages else 'en'
        
        # Rapid preprocessing
        clean_text = text.strip()
        
        # Direct translation with minimal processing
        translation = translate_text(clean_text, source_lang, target_lang)
        
        return {
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
    
    except Exception as e:
        logger.error(f"Fast translation error: {e}")
        return {
            "translation": text,  # Fallback to original text
            "error": str(e)
        }

# Optimized Batch Translation
def rapid_batch_translate(
    texts: List[str], 
    source_lang: str = None, 
    target_lang: str = 'en'
):
    """
    Concurrent batch translation with minimal overhead
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Parallel translation processing
        futures = [
            executor.submit(
                fast_translation_processor, 
                text, 
                source_lang, 
                target_lang
            ) 
            for text in texts
        ]
        
        # Collect translations as they complete
        translations = [
            future.result()['translation'] 
            for future in concurrent.futures.as_completed(futures)
        ]
    
    return translations

# Lightweight Translation Endpoint
@app.route('/rapid_translate', methods=['POST'])
def rapid_translation_endpoint():
    """
    Ultra-lightweight translation endpoint
    """
    data = request.get_json()
    texts = data.get('texts', [data.get('text')])
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang', 'en')
    
    if not texts:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Single text or batch translation
        if len(texts) == 1:
            result = fast_translation_processor(
                texts[0], 
                source_lang, 
                target_lang
            )
        else:
            result = {
                "translations": rapid_batch_translate(
                    texts, 
                    source_lang, 
                    target_lang
                )
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Streaming Translation Support
@app.route('/stream_translate', methods=['POST'])
def streaming_translation_endpoint():
    """
    Real-time streaming translation endpoint
    """
    def generate_stream():
        try:
            data = request.get_json()
            text = data.get('text', '')
            source_lang = data.get('source_lang')
            target_lang = data.get('target_lang', 'en')
            
            # Tokenize and stream translation
            inputs = tokenizer(text, return_tensors="pt")
            
            for token in model.generate(
                inputs.input_ids, 
                max_length=len(inputs.input_ids[0]) + 50,
                num_return_sequences=1
            ):
                chunk = tokenizer.decode(token, skip_special_tokens=True)
                yield f"data: {chunk}\n\n"
        
        except Exception as e:
            yield f"error: {str(e)}"
    
    return Response(generate_stream(), mimetype='text/event-stream')

# Zero-Copy Translation Mechanism
def zero_copy_translate(text: str, source_lang: str = None, target_lang: str = 'en'):
    """
    Translation with minimal memory overhead
    """
    # Use shared memory for efficient transfer
    translation = fast_translation_processor(text, source_lang, target_lang)
    return torch.tensor(translation['translation'].encode()).share_memory_()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug to True for detailed error messages
