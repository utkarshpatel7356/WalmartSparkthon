import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Hello!")
print(response.text)
