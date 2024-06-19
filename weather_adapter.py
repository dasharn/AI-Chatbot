import requests
from configparser import ConfigParser


class WeatherAdapter:
	"""
    A class to interact with the OpenWeatherMap API to fetch weather data.

    This adapter class is designed to provide an interface for fetching weather information
    from the OpenWeatherMap API. It requires an API key for authentication and uses a predefined
    base URL to make requests for current weather data.

    Attributes:
        api_key (str): The API key used for authenticating requests to the OpenWeatherMap API.
        base_url (str): The base URL for the OpenWeatherMap API's current weather data endpoint.

    Parameters:
        api_key (str): An API key provided by the user for accessing the OpenWeatherMap API.
    """
	def __init__(self, api_key):
		self.api_key = api_key
		self.base_url = "http://api.openweathermap.org/data/2.5/weather"

	def _get_api_key(self):
		"""Fetch the API key from your configuration file.

		Expects a configuration file named "secrets.ini" with structure:

			[openweather]
			api_key=<YOUR-OPENWEATHER-API-KEY>
		"""
		config = ConfigParser()
		config.read("secrets.ini")
		return config["openweather"]["api_key"]


	def get_weather(self, city_name):
		"""
		Fetches the current weather description for a specified city.

		This method sends a request to the OpenWeatherMap API using the provided city name and API key. It extracts and returns the weather description from the API response. If the request fails, it prints an error message and returns None.

		Parameters:
		- city_name (str): The name of the city for which to fetch the weather.

		Returns:
		- str: The weather description for the specified city if the request is successful; otherwise, None.
		"""
		api_url = f"{self.base_url}?q={city_name}&appid={self.api_key}"
		response = requests.get(api_url)
		if response.status_code == 200:
			response_dict = response.json()
			weather = response_dict["weather"][0]["description"]
			return weather
		else:
			print(f'[!] HTTP {response.status_code} calling [{api_url}]')
			return None
		

