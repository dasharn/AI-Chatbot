# AI ChatBot 

This repository contains a Python-based conversational AI chatbot designed to interact with users through speech and text. The chatbot leverages various AI and NLP techniques to perform tasks such as fetching weather information, providing the current time, responding to common phrases, and engaging in general conversation.

## Components

- **Speech Recognition**: Converts speech captured from the microphone into text using Google's Speech Recognition API.
- **Text-to-Speech**: Converts text responses into speech using the gTTS (Google Text-to-Speech) library, ensuring a seamless user experience.
- **Conversational AI**: Uses the DialoGPT-medium model from Hugging Face for generating human-like responses during conversations.
- **Weather Information**: Fetches real-time weather information for specified cities using the OpenWeatherMap API.
- **Error Handling**: Logs errors and exceptions to `chatbot_errors.log`, ensuring robustness and easy debugging.

## Setup

### Dependencies

Install required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Classes

### ChatBot Class

- **Initialization**: Sets up necessary models (e.g., DialoGPT for conversation, SentenceTransformer for similarity), initializes attributes, and loads configurations.

### WeatherAdapter Class

- **Description**: Interacts with the OpenWeatherMap API to fetch weather data for specified cities.

## Supporting Libraries

- **External Libraries**: Utilizes SpeechRecognition for speech-to-text conversion, gTTS for text-to-speech, transformers and sentence-transformers for NLP tasks, spaCy for city recognition, and numpy for random message selection.
- **Internal Libraries**: Integrates a logging system to record errors, ensuring robust error handling and debugging capabilities.

## Usage

1. **Setup**
   - Ensure all dependencies (`speech_recognition`, `gtts`, `transformers`, `sentence_transformers`, `spacy`, `pyttsx3`) are installed.
   - Obtain an API key from OpenWeatherMap and configure it in the `secrets.ini` file under the `[openweather]` section.

2. **Execution**
   - Run the `ChatBot` class by executing `python chatbot.py`.
   - Follow the prompts to interact with the chatbot using speech or text.

## Error Handling

- Errors and exceptions are logged to `chatbot_errors.log` for debugging purposes.
- Graceful error handling ensures smooth operation even in case of speech recognition failures or API request issues.

## Notes

- The chatbot is designed for single-task interactions due to limitations in handling multiple requests simultaneously.
- Enhancements and modifications can be made to support additional functionalities or improve existing ones based on specific requirements.

## Contributors

- Developed by Dash
