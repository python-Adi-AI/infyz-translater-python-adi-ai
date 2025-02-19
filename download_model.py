from transformers import MarianMTModel

# Define model name
model_name = "Helsinki-NLP/opus-mt-en-fr"

# Download and save the model
model = MarianMTModel.from_pretrained(model_name)
model.save_pretrained("./model")

print("Model downloaded and saved in the 'model' folder.")


