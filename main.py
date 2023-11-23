from flask import Flask, render_template, request, jsonify, send_file
import openai
from google.cloud import translate_v2 as translate
import azure.cognitiveservices.speech as speechsdk
import os
import logging
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# OpenAI API configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    openai.api_key = openai_api_key
else:
    logging.error('OpenAI API key not found. Please set it in your environment variables.')

# Google Cloud Translate client configuration
google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if google_credentials_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
    translate_client = translate.Client()
else:
    logging.error('Google Cloud credentials path not found. Please set it in your environment variables.')

# Azure Speech Service configuration
speech_key = os.getenv('AZURE_SPEECH_KEY')
service_region = os.getenv('AZURE_SERVICE_REGION')
if speech_key and service_region:
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = "en-US"
    speech_config.speech_synthesis_language = "en-US"
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
else:
    logging.error('Azure Speech credentials not found. Please set them in your environment variables.')

# Initialize conversation history
conversation_history = []

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('user_input')

    # Get response from OpenAI based on conversation history and user input
    openai_response = ask_openai(conversation_history, user_input)
    if not openai_response:
        return jsonify({'response': 'OpenAI request failed'})

    # Translate response back to the original language
    translated_response = translate_text(openai_response, target_language='ka')  # Replace 'ka' with the desired language code
    if not translated_response:
        return jsonify({'response': 'Translation failed'})

    # Convert text response to speech using Azure TTS
    audio_file_path = text_to_speech(translated_response)

    return jsonify({'response': translated_response, 'audio_url': audio_file_path})


@app.route('/audio/<path:filename>')
def audio(filename):
    return send_file(filename, mimetype="audio/wav")

def text_to_speech(text):
    try:
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        audio_file_path = "audio_output.wav"  # Define your audio file path here
        audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_file_path)
        result = speech_synthesizer.speak_text_async(text, audio_config=audio_config).get()

        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logging.error(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logging.error(f"Error details: {cancellation_details.error_details}")
            return None
        return audio_file_path
    except Exception as e:
        logging.error(f"Error in text_to_speech: {e}")
        return None

def translate_text(text, target_language='en'):
    try:
        translation = translate_client.translate(text, target_language=target_language)
        return translation['translatedText']
    except Exception as e:
        logging.error(f"Error in translate_text: {e}")
        return None



def ask_openai(conversation_history, user_input):
    try:
        # Translate user input to English
        translated_input = translate_text(user_input, target_language='en')
        if not translated_input:
            return None

        # Add user input to the conversation history
        conversation_history.append({"role": "user", "content": translated_input})

        # Generate a response from OpenAI based on the updated conversation history
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=conversation_history,
        )

        # Extract and format the response content
        response_text = response.choices[0].message.content.strip()

        # Return the response text
        return response_text

    except Exception as e:
        logging.error(f"Error in ask_openai: {e}")
        return None

if __name__ == '__main__':
  app.run(port=5000)
