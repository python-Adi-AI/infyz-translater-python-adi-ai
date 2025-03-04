from transformers import MarianMTModel

def download_translation_model(model_name="Helsinki-NLP/opus-mt-en-fr", save_path="./model"):
    """
    Download and save a pre-trained translation model.
    
    :param model_name: Hugging Face model identifier
    :param save_path: Local directory to save the model
    """
    try:
        model = MarianMTModel.from_pretrained(model_name)
   
        model.save_pretrained(save_path)
        print(f"Model '{model_name}' downloaded and saved in '{save_path}'.")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_translation_model()
