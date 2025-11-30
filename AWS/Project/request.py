import requests
import json

url = "https://22uuwmqt6i.execute-api.us-east-1.amazonaws.com/dev/research/"

payload = json.dumps({"topic": "AI in health care in 7 word"})
headers = {"Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
