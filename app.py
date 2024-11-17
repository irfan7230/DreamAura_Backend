from flask import Flask, request, jsonify
from flask_cors import CORS
from g4f.client import Client  # GPT-4o
from g4f.cookies import set_cookies
import firebase_admin
from firebase_admin import credentials, db
import os
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize
import time
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
client = Client()

CORS(app)

logging.basicConfig(level=logging.INFO)

nltk.download('punkt', quiet=True)

MAX_ATTEMPTS = 5
RETRY_DELAY = 1  # seconds

# Initialize Firebase
cred = credentials.Certificate('./configs/dreamweb-6acb3-firebase-adminsdk-8lt7y-a89b4c42b9.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://dreamweb-6acb3-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

set_cookies(".google.com", {
    "__Secure-1PSID": "g.a000owhLQs4tTefHn9UveLoTITXX9BcykGHI700RaUo7OQGFm1oNKbKq7Dm_ccatmDyIbDBf7AACgYKAacSARcSFQHGX2MiNZtKcO6IBBbEqu3aZO8fcBoVAUF8yKreCExQ2xSeJCWFlG1__8QW0076",
    "SNlM0e": "AGUzedNby2xgNDy3otViWl6W-_5R:1728899853659"
})

set_cookies(".bing.com", {
    "_U": "17N-aX_OLckfLRoio20oYGN4z2L9GW9UHpax3UrEoXN56NvY5m9sAXVyuYPhCW-XDsbu0vpeu5TBRvPxQoyKuskj6W8SQaNEgOj1H_1DtFCLfOP2NrmtPMlX1FyqCBel1OJmpwCjbkSbVixNZh2Z151HT1bCZcUYKT-QWAeBe2b5nFJzIgdzf9SryQZB8A1L3zRCq9LAN_ti7j67p2kYMmv0t9IddfJxRg069OU45c8o"
})

def get_dream_text(data):
    return data.get('dream')

def clean_raw_response(raw_response):
    cleaned_response = raw_response.replace("**", "").replace("#"," ").replace("\r", " ").strip()
    return cleaned_response

def get_gpt4o_response(dream_text):
    prompt = f"""
    1. Interpret this dream: {dream_text}. Provide a detailed analysis that includes the following components:
        - **Summary:** Provide the summary of user dream.
        - **Psychological Meaning:** Provide insights on the psychological aspects related to this dream.
        - **Mental Health Status:** Assess the user's mental health based on the dream's content.
        - **Physical Health Insights:** Highlight any physical health insights connected to the dream.
        - **Disease Indications:** Mention any potential disease indications inferred from the dream.
        - **Dream Convey Message:** Explain what this dream conveys about the user's life or emotions.
        - **Sleep Quality Suggestions:** Offer suggestions on improving sleep quality based on the dream.
        - **Lucid Dream:** Indicate whether this dream is considered a lucid dream or not.
    
    2. Based on this dream, predict the next 2 upcoming dreams the user might have. For each dream, provide the following details in the specified format:
        - **Dream 1:**
            - **Theme or Title:** [Insert Title Here]
            - **Predicted Date:** [Insert the dream will be coming at predictable date mind it you(gpt) can predict that based on dream prompt mind it the predicted date will be atleast above 2024 oct]
            - **Description:** [Insert Description Here]
        - **Dream 2:**
            - **Theme or Title:** [Insert Title Here]
            - **Predicted Date:** [Insert the dream will be coming at predictable date mind it you(gpt) can predict that based on dream prompt mind it the predicted date will be atleast above 2024 oct]
            - **Description:** [Insert Description Here]
    
    predict the approximate date for upcoming dreams
"""


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    if hasattr(response, 'choices') and len(response.choices) > 0:
        generated_text = response.choices[0].message.content.strip()  
        cleaned_text = clean_raw_response(generated_text)  
        print("Generated Text:", cleaned_text)
        return cleaned_text
    return ""


def parse_response(cleaned_text):
    sentences = sent_tokenize(cleaned_text)

    output = {
        "summary": "",
        "psychological_meaning": {
            "heading": "Psychological Meaning",
            "insights": []
        },
        "mental_health_status": {
            "heading": "Mental Health Status",
            "insights": []
        },
        "physical_health": {
            "heading": "Physical Health",
            "insights": []
        },
        "disease_indications": {
            "heading": "Disease Indications",
            "insights": []
        },
        "dream_convey_message": {
            "heading": "Dream Convey Message",
            "insights": []
        },
        "sleep_quality": {
            "heading": "Sleep Quality",
            "insights": []
        },
        "lucid_dream": {
            "heading": "Lucid Dream",
            "insights": []
        },
        "additional_resources": {
            "heading": "Additional Resources",
            "links": [
                "https://www.mentalhealth.org.uk/your-mental-health/dreams-and-nightmares",
                "https://www.healthline.com/health/dreams-about-falling"
            ]
        },
        "upcoming_dreams": []
    }

    section_headers = {
        "summary": "Summary:",
        "psychological_meaning": "Psychological Meaning:",
        "mental_health_status": "Mental Health Status:",
        "physical_health": "Physical Health Insights:",
        "disease_indications": "Disease Indications:",
        "dream_convey_message": "Dream Convey Message:",
        "sleep_quality": "Sleep Quality Suggestions:",
        "lucid_dream": "Lucid Dream:"
    }

    current_section = None
    for sentence in sentences:
        sentence = sentence.strip()

        # Check for new section header
        for key, header in section_headers.items():
            if header.lower() in sentence.lower():
                current_section = key
                sentence = sentence.replace(header, "").strip() 
                break

        if current_section:
            if current_section == "summary":
                output["summary"] += sentence
            else:
                output[current_section]["insights"].append(sentence)

    
    # Extract upcoming dreams
    upcoming_dreams_start = cleaned_text.find("Dream 1:")
    if upcoming_dreams_start != -1:
        upcoming_dreams_text = cleaned_text[upcoming_dreams_start:]

        lines = upcoming_dreams_text.splitlines()
        upcoming_dream = {}
        for line in lines:
            line = line.strip()
            if line.startswith("Dream"):
                if upcoming_dream:  
                    output["upcoming_dreams"].append(upcoming_dream)
                upcoming_dream = {"title": None, "predicted_date": None, "description": None}
                
            if "- Theme or Title:" in line:
                upcoming_dream["title"] = line.split(":")[1].strip().strip('"')
            elif "- Predicted Date:" in line:
                upcoming_dream["predicted_date"] = line.split(":")[1].strip()
            elif "- Description:" in line:
                upcoming_dream["description"] = line.split(":")[1].strip()

        if upcoming_dream and (upcoming_dream["title"] or upcoming_dream["predicted_date"]):
            output["upcoming_dreams"].append(upcoming_dream)

    if not output["upcoming_dreams"]:
        output["upcoming_dreams"].append({"title": None, "predicted_date": None, "message": "No upcoming dreams are predicted at this time."})
        
    return output


@app.route('/submit_dream', methods=['POST'])
@retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_DELAY))
def interpret_dream():
    try:
        data = request.json
        dream_text = get_dream_text(data)
        dreamer_id = data.get('dreamerID')
    
        if not dream_text:
            logging.error("No dream text provided")
            return jsonify({"error": "No dream text provided"}), 400
        if not dreamer_id:
            logging.error("No dreamerID provided")
            return jsonify({"error": "No dreamerID provided"}), 400

        logging.info(f"Received dream text: {dream_text}")
        print(f"Dreamer ID: {dreamer_id}")

        raw_response = get_gpt4o_response(dream_text)
        parsed_response = parse_response(raw_response)

        logging.info(f"Parsed GPT-4o response: {parsed_response}")

        upcoming_dreams = parsed_response.get('upcoming_dreams', [])

        if upcoming_dreams:
            user_ref = db.reference(f'users/{dreamer_id}/upcoming_dreams')

            print(f"Firebase reference: users/{dreamer_id}/upcoming_dreams")

            try:
                user_ref.delete()

                for dream in upcoming_dreams:
                    if dream.get('title') and dream.get('predicted_date'):
                        dream_data = {
                            'title': dream.get('title'),
                            'predicted_date': dream.get('predicted_date')
                        }

                        user_ref.push(dream_data)
                        print(f"Stored upcoming dream in Firebase: {dream_data}")
                    else:
                        print(f"Skipping incomplete dream: {dream}")
            except Exception as firebase_error:
                logging.error(f"Error storing dreams in Firebase: {firebase_error}")
                return jsonify({"error": "Failed to store dreams in Firebase"}), 500
        else:
            print("No upcoming dreams to store")

        return jsonify(parsed_response), 200

    except Exception as e:
        logging.error(f"Error interpreting dream: {e}")
        return jsonify({"error": "An error occurred during interpretation"}), 500

    
@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    dream_text = get_dream_text(data)
    gender = data.get('gender')
    
    try:
        response = client.images.generate(
            model="playground-v2.5",
            prompt="I am a "+gender+" in my dream, "+dream_text
        )
        
        if response.data and len(response.data) > 0:
            image_url = response.data[0].url 
            if image_url:
                print(f"Generated image URL: {image_url}")
            else:
                print("No URL found in the response.")
        else:
            print("No image data found in the response.")
        
        return jsonify({"image_url": image_url})
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
