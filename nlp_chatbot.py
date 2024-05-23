import os
import json
from flask import Flask, render_template, request, jsonify, Response
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize  # Tokenization
from nltk.tag import pos_tag  # POS tagging
import re  # Regular Expressions

# Initialize Flask app
app = Flask(__name__)

# Configure Generative AI
API_KEY = 'AIzaSyAb4CKQ23uIo9PH-FwkGmoB3yHoJHaYuOI'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

nltk.download('punkt')  # Download tokenizer data
nltk.download('averaged_perceptron_tagger')  # Download POS tagger data

# Store session states in a dictionary (for simplicity, in production consider using a more robust solution)
sessions = {}

@app.route('/')
def root():
    return render_template('nlp_chatbot.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    session_id = request.json.get('session_id', 'default')
    user_message = request.json['message']
    
    if session_id not in sessions:
        sessions[session_id] = {'state': 'ask_cuisine'}

    state = sessions[session_id]['state']
    response_text = ""
    show_buttons = False
    buttons = []
    dish_name = ""

    try:
        if state == 'ask_cuisine':
            response_text = "Which cuisine do you prefer?"
            buttons = ["Filipino", "Chinese", "Korean", "Japanese", "Thai", "Indian", "French", "Brazilian", "Mexican"]
            show_buttons = True
            sessions[session_id]['state'] = 'ask_ingredients'
        elif state == 'ask_ingredients':
            if user_message not in ["Filipino", "Chinese", "Korean", "Japanese", "Thai", "Indian", "French", "Brazilian", "Mexican"]:
                response_text = "Please choose a valid cuisine from the options provided."
                show_buttons = True
                buttons = ["Filipino", "Chinese", "Korean", "Japanese", "Thai", "Indian", "French", "Brazilian", "Mexican"]
                return jsonify({'message': response_text, 'show_buttons': show_buttons, 'buttons': buttons})
            sessions[session_id]['cuisine'] = user_message
            response_text = f"You selected {user_message}. Please enter the ingredients you have."
            sessions[session_id]['state'] = 'generate_recipes'
        elif state == 'generate_recipes':
            cuisine = sessions[session_id].get('cuisine', 'general')
            ingredients = validate_ingredients(user_message)
            if not ingredients:
                response_text = "Please provide valid ingredients (e.g., chicken, rice, garlic)."
            else:
                ingredients_str = ", ".join(ingredients)
                prompt = f"Generate a {cuisine} dish using the following ingredients: {ingredients_str}. Provide the recipe name followed by ingredients, instructions, nutritional information, common allergens, and possible substitutions."
                response = chat.send_message(prompt)
                response_text = response['message']
                filtered_dishes = filter_dish_lines_with_regex(response_text)
                if filtered_dishes:
                    dish_name = filtered_dishes[0].split('. ')[1]
                    response_text = format_dish_details(response_text)
                sessions[session_id]['state'] = 'ask_cuisine'
    except Exception as e:
        response_text = f"An error occurred: {str(e)}"
        sessions[session_id]['state'] = 'ask_cuisine'

    return jsonify({'message': response_text, 'show_buttons': show_buttons, 'buttons': buttons, 'dish_name': dish_name})

def validate_ingredients(ingredients_text):
    ingredients = ingredients_text.split(',')
    valid_ingredients = []
    for ingredient in ingredients:
        ingredient = ingredient.strip().lower()
        tokens = word_tokenize(ingredient)
        pos_tags = pos_tag(tokens)
        if any(tag.startswith('NN') for word, tag in pos_tags):  # Ensure the word is a noun (NN).
            valid_ingredients.append(ingredient)
    return valid_ingredients

def filter_dish_lines_with_regex(text):
    """
    Filtering with Regular Expressions (Regex)
    - Using regex to filter dish recommendations that start with a digit followed by a period.
    """
    lines = text.split('\n')
    filtered_lines = [line.strip() for line in lines if re.match(r'^\d+\. ', line.strip())]
    return filtered_lines

def format_dish_details(text):
    """
    Text Preprocessing
    - Formatting dish details for better display.
    """
    sections = text.split("\n\n")
    formatted_text = ""
    for section in sections:
        if section.startswith("Ingredients"):
            formatted_text += f"<strong>{section.split(':')[0]}:</strong><br>{section.split(':')[1].strip().replace('\n', '<br>')}"
        elif section.startswith("Instructions"):
            formatted_text += f"<strong>{section.split(':')[0]}:</strong><br>{section.split(':')[1].strip().replace('\n', '<br>')}"
        elif section.startswith("Nutritional Information"):
            formatted_text += f"<strong>{section.split(':')[0]}:</strong><br>{section.split(':')[1].strip().replace('\n', '<br>')}"
        elif section.startswith("Common Allergens"):
            formatted_text += f"<strong>{section.split(':')[0]}:</strong><br>{section.split(':')[1].strip().replace('\n', '<br>')}"
        elif section.startswith("Possible Substitutions"):
            formatted_text += f"<strong>{section.split(':')[0]}:</strong><br>{section.split(':')[1].strip().replace('\n', '<br>')}"
        else:
            formatted_text += section.strip().replace('\n', '<br>')
        formatted_text += "<br><br>"
    return formatted_text

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
