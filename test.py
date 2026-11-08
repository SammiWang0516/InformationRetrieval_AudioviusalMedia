import json
import os

import requests
from dotenv import load_dotenv


# 1. Read the token from .env
load_dotenv(".env")
token = os.getenv("TMDB_READ_TOKEN")

# 2. Choose an API endpoint
url = "https://api.themoviedb.org/3/movie/11"

# 3. Call the API
response = requests.get(
    url,
    headers={"Authorization": f"Bearer {token}"},
    params={"language": "en-US"},
    timeout=20,
)

response.raise_for_status()

# 4. Convert the JSON response into a Python dictionary
data = response.json()

# Show which fields are available
print(data.keys())

# Show the complete response
print(json.dumps(data, indent=2, ensure_ascii=False))