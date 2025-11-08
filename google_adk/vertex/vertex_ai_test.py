from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Hello from Vertex AI!")
print(response.text)
