import os
import platform
import tempfile
import time
import datetime
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from transformers import pipeline, Conversation
from sentence_transformers import SentenceTransformer, util
import spacy # For finding cities
import logging
import pyttsx3
import weather_adapter as WeatherAdapter

logging.basicConfig(filename='chatbot_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class ChatBot():
    """
    A conversational agent (chatbot) that can engage in dialogue and perform specific tasks based on user input.

    Attributes:
        name (str): The name of the chatbot.
        text (str): The latest message received from the user.
        chat (pipeline): A Hugging Face pipeline for conversational tasks using the DialoGPT-medium model.
        similarity_model (SentenceTransformer): A model for computing sentence embeddings and similarities.
        city_identifier (Language): A spaCy model for processing English text, used for identifying city names.
        exit_flag (bool): A flag to indicate whether the chatbot should terminate its process.
        expected_inputs (dict): A mapping of expected input phrases to their corresponding method calls.
        weather (WeatherAdapter): An adapter for fetching weather information.

    The chatbot initializes with a given name, sets up necessary models for conversation and task execution,
    and prepares a set of expected inputs with their associated actions. It uses the DialoGPT-medium model for
    generating conversational responses, the SentenceTransformer 'stsb-roberta-large' for sentence similarity,
    and a spaCy model for city name recognition. The WeatherAdapter is used to fetch weather information.
    """
    def __init__(self, name):
        print(f"<<<< Starting up {name} >>>>")
        self.name = name
        self.text = ""
        self.chat = pipeline("conversational", model="microsoft/DialoGPT-medium")
        self.similarity_model = SentenceTransformer('stsb-roberta-large')
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.city_identifier = spacy.load("en_core_web_sm")
        self.exit_flag = False
        self.expected_inputs = {
            "Current Weather in the city":self.get_weather(),
            "What is the time?":self.get_time(),
            "Thank you": self.thank_you(),
            "Close Down": self.sleep(),
            "Wake Up {self.name}": self.wake_up()
        }
        self.exit_flag = False
        self.weather = WeatherAdapter()


    def speech_to_text(self):
        """
        Converts speech captured from the microphone into text using Google's Speech Recognition API.

        This method listens for speech through the default microphone, attempts to recognize the speech,
        and then assigns the recognized text to the `self.text` attribute. If the speech is not recognized,
        or if there are any errors in the speech recognition process (such as a request error to the Google API),
        `self.text` is set to "ERROR", and the error is logged.

        Uses:
            - SpeechRecognition library's Recognizer and Microphone for capturing and recognizing speech.
            - Google's Speech Recognition API for converting speech to text.

        Side Effects:
            - Modifies `self.text` with the recognized text or "ERROR" if an error occurs.
            - Prints the recognized text to the console.
            - Logs an error message if speech recognition fails or an error occurs.
        """
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as mic:
                print("Listening...")
                audio = recognizer.listen(mic)
                self.text = recognizer.recognize_google(audio)
                print("Me  >>> ", self.text)
        except sr.UnknownValueError:
            self.text = "ERROR"
            logging.error("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            self.text = "ERROR"
            logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            self.text = "ERROR"
            logging.error(f"An error occurred: {e}")

    
    
    def text_to_speech(self, text):
        """
        Converts the given text into speech and plays it through the system's default audio output.

        This method uses the gTTS (Google Text-to-Speech) library to convert the provided text into speech,
        saving the audio to a temporary file. It then plays this audio file using the default media player
        for the system's operating system. The method handles different commands for playing audio files
        on Windows, macOS (Darwin), and Unix/Linux systems. After playing the audio, it attempts to remove
        the temporary file.

        Parameters:
            text (str): The text to be converted into speech.

        Side Effects:
            - Outputs the converted speech to the system's default audio output.
            - Creates and then deletes a temporary audio file in the system's temporary directory.
            - Prints a message to the console if an error occurs during the text-to-speech conversion
            or while removing the temporary file.
        """
        print(f"{self.name} --> ", text)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                speaker = gTTS(text=text, lang="en", slow=False)
                speaker.save(fp.name)
                # Determine the platform and play the audio file accordingly
                if platform.system() == "Windows":
                    os.system(f'start {fp.name}')
                elif platform.system() == "Darwin":  # macOS
                    os.system(f'afplay {fp.name}')
                else:  # Assume Linux or other Unix
                    os.system(f'mpg123 {fp.name}')
                # Wait for the speech to finish based on an estimated duration
                time.sleep(len(text) / 10)  # Adjust as needed for accuracy
        except Exception as e:
            print(f"An error occurred during text-to-speech conversion: {e}")
        finally:
            try:
                os.remove(fp.name)  # Clean up the temporary file
            except OSError as e:
                print(f"Error removing temporary audio file: {e}")

    def wake_up(self, text):
        """
        Generates a greeting message introducing the chatbot and prompting the user for a single request.

        This method constructs a greeting message that includes the chatbot's name and a brief instruction
        for the user to ask for one thing at a time, acknowledging the chatbot's limitations in handling
        multiple requests simultaneously. It is designed to be called when the chatbot is initiated or
        "woken up" by the user.

        Parameters:
            text (str): The text input from the user, which triggers the chatbot to wake up. Currently,
                        this parameter is not used in the method, but it is included for potential future
                        enhancements where the wake-up command might be customized or include additional
                        instructions.

        Returns:
            str: A greeting message from the chatbot, including its name and a prompt for the user to
                 make a single request.
        """
        return f"Hello I am {self.name} your AI assistant. Please ask for one thing at a time, I'm not all that smart. What can I do for you?"
    
    def sleep(self, text):
        """
        Sets the chatbot's exit flag to True and returns a random farewell message.

        This method is intended to be called when the user wants to end the conversation with the chatbot.
        It sets the `exit_flag` attribute to True, indicating that the chatbot should terminate its process.
        Additionally, it selects a random farewell message from a predefined list to make the goodbye more
        personable.

        Parameters:
            text (str): The text input from the user, which triggers the chatbot to prepare for shutdown.
                        Currently, this parameter is not used in the method, but it is included for potential
                        future enhancements where the shutdown command might be customized.

        Returns:
            str: A randomly selected farewell message.
        """
        self.exit_flag = True
        return np.random.choice([
            "Cheerio", 
            "Ta-ta", 
            "Pip-Pip", 
            "Take care, old chap", 
            "Until next time", 
            "Ok, off you pop"
        ])
        
    
    def thank_you(self):
        """
        Returns a random expression of gratitude.

        This method is designed to respond to the user's thanks with a variety of polite and friendly
        expressions of gratitude. It selects a random message from a predefined list to provide some
        variability in the chatbot's responses, making the interaction feel more dynamic and less
        repetitive.

        Returns:
            str: A randomly selected expression of gratitude.
        """
        return np.random.choice([
            "You're welcome!", 
            "Anytime!", 
            "No problem!", 
            "Cheers!", 
            "All good!", 
            "No worries!", 
            "Don't mention it!", 
            "It's nothing!", 
            "My pleasure!", 
            "Happy to help!"
        ])

    def get_weather(self):
      """
      Retrieves the weather information for a specified city from the user's input text.
  
      This method processes the user's input text to identify a city name using the `city_identifier`
      model, which is expected to be a spaCy model loaded with the capability to recognize geopolitical
      entities (GPE). If a city name is identified, it attempts to fetch the weather information for
      that city using the `weather` adapter's `get_weather` method.
  
      If the method successfully identifies a city and retrieves weather information, it returns a
      formatted string describing the weather in that city. If no city is identified in the input text,
      or if the weather information cannot be retrieved, it returns a polite error message.
  
      Returns:
          str: A message containing the weather information for the identified city, or an error
              message if the city cannot be identified or if weather information cannot be retrieved.
      """
      statement = self.city_identifier(self.text)
  
      city = None
      for ent in statement.ents:
          if ent.label_ == "GPE": # GeoPolitical Entity
              city = ent.text
              break
  
      if city:
          city_weather = self.weather.get_weather(city)
          if city_weather is not None:
              return f"Ah, in {city}, one observes the weather to be {city_weather}. Quite intriguing, wouldn't you agree?"
          else:
              return "My apologies, but I regret to inform you that I was unable to procure the weather information for your request."
      else:
          return "I say, old chap, one must specify a city to inquire about the weather."


    
    def get_time(self):
        return datetime.datetime.now().strftime('%H:%M')

    def handle_conversation(self):
        """
        Handles the conversation flow by processing the user's input and generating an appropriate response.

        This method first checks if the user's input text is "ERROR", indicating a failure in speech recognition,
        and returns a prompt for the user to repeat their input. If the input is valid, it computes the similarity
        between the user's input and a list of expected inputs using sentence embeddings. If a high similarity
        (above a threshold of 0.7) is found with any of the expected inputs, it triggers the corresponding action
        associated with the best match.

        If no expected input matches closely enough, the method defaults to using a conversational model (`self.nlp`)
        to generate a response based on the user's input. This response is then extracted from the model's output
        and returned.

        Returns:
            str: The generated response to the user's input, either by triggering a predefined action or by
                using the conversational model.
        """
        if self.text == "ERROR":
            return "Sorry, come again?"
        
        max_similarity, best_match = -1, None

        expected_input_list = [self.text] + list(self.expected_inputs.keys())
        expected_input_embeddings = self.similarity_model.encode(expected_input_list, convert_to_tensor=True)

        for i in expected_input_embeddings[1:]:
            similarity = util.pytorch_cos_sim(expected_input_embeddings[0], expected_input_embeddings[i])
            if similarity > max_similarity:
                max_similarity, best_match = similarity, expected_input_list[i]

        if max_similarity > 0.7:
            return self.expected_inputs[best_match]
        
        chat = self.nlp(Conversation(self.text), pad_token_id=50256)
        res = str(chat)
        return res[res.find("bot >> ") + 6:].strip()

    def run(self):
        """
        Initiates and manages the main loop of the chatbot's operation.

        This method sets the `exit_flag` to False and enters a loop where it continuously listens for
        user input through speech, processes the input to generate a response, and then converts this
        response back into speech. The loop continues until the `exit_flag` is set to True, indicating
        that the user has requested to end the conversation or the chatbot has been otherwise instructed
        to shut down. Upon exiting the loop, a closing message is printed to indicate that the chatbot
        is shutting down.

        The main steps within the loop include:
        1. Converting speech to text.
        2. Handling the conversation based on the converted text to generate a response.
        3. Converting the response text back into speech.

        Side Effects:
            - Modifies `self.exit_flag` based on user commands or internal conditions.
            - Prints a closing message to the console upon shutdown.
        """
        self.exit_flag = False
        while not self.exit_flag:
            self.speech_to_text()
            response = self.handle_conversation()
            self.text_to_speech(response)
        print(f"<<<< Closing down {self.name} >>>>")

# Running the AI
if __name__ == "__main__":
    chatbot = ChatBot("Jarvis")
    chatbot.run()
