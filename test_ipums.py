import os
from dotenv import load_dotenv
from ipumspy import IpumsApiClient

# Load the .env file
load_dotenv()

# Get API key
API_KEY = os.getenv("IPUMS_API_KEY")

# Create client
client = IpumsApiClient(api_key=API_KEY)

print("✅ IPUMS client connected successfully!")
