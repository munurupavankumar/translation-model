from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer
import pytesseract
from PIL import Image

app = Flask(__name__)
CORS(app)

# Initialize translation model
model_name = 'Helsinki-NLP/opus-mt-en-fr'  # Example for English to French
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Endpoint for text translation
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data['text']
    source_lang = data['source_lang']
    target_lang = data['target_lang']

    # Dynamically set model based on source and target language
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    tokenized_text = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return jsonify({'translated_text': translated_text[0]})


# Endpoint for document OCR and translation
@app.route('/translate-document', methods=['POST'])
def translate_document():
    # Retrieve the source and target languages from the form data
    source_lang = request.form['source_lang']
    target_lang = request.form['target_lang']

    # Retrieve the uploaded document
    image = request.files['document']
    img = Image.open(image)

    # Perform OCR to extract text
    text = pytesseract.image_to_string(img)

    # Load the correct model based on source and target languages
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Translate the extracted text
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    # Return the translated text as a response
    return jsonify({'translated_text': translated_text[0]})


if __name__ == '__main__':
    app.run(debug=True)
