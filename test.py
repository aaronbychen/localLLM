from dotenv import load_dotenv
import os

load_dotenv()

# printing environment variables
API_KEY = os.getenv("API_KEY")
print(API_KEY)