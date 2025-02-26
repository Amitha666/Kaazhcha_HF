from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, 
    AutoProcessor, AutoModelForCausalLM
)
import io

app = Flask(__name__)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)
git_processor = AutoProcessor.from_pretrained("microsoft/git-base-textcaps")
git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)

def analyze_image(image, question):
    """Analyze image and generate response based on query."""
    
    if question.lower().startswith(("is", "does", "can")):
        # Yes/No type question → Use BLIP-VQA
        formatted_question = (
            f"Look at the image carefully. Question: {question}. "
            f"Answer strictly with either 'Yes' or 'No'. If unclear, say 'Not sure'."
        )

        inputs = blip_processor(image, formatted_question, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs, max_length=40)
        response = blip_processor.decode(output[0], skip_special_tokens=True).strip().lower()

        if "yes" in response:
            return "Yes"
        elif "no" in response:
            return "No"
        elif "not sure" in response or response == "":
            return "I'm not sure."
        else:
            return "I'm not confident in the answer."

    else:
        # Open-ended question → Use GIT for descriptions
        inputs = git_processor(images=image, return_tensors="pt").to(device)
        output = git_model.generate(**inputs, max_length=50)
        response = git_processor.decode(output[0], skip_special_tokens=True).strip()

        if len(response.split()) < 5:
            return "I'm unable to provide details about this image."
        
        return response

@app.route('/process', methods=['POST'])
def process_request():
    """Handles requests from FlutterFlow."""
    try:
        # Get image and prompt from request
        if 'image' not in request.files or 'prompt' not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_file = request.files['image']
        user_prompt = request.form['prompt']

        # Open image
        image = Image.open(io.BytesIO(image_file.read()))

        # Process image and prompt
        response = analyze_image(image, user_prompt)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
