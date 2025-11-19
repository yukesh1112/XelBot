import google.generativeai as genai

# Final test with the correct model
API_KEY = "AIzaSyA0ePTaWI-Up6oTuG_R_B-Pvave0UwumjM"

print("🎯 Final API Test")
print("=" * 20)

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    
    response = model.generate_content("Hello! Please respond with 'XelBot is ready to work!' to confirm the API is working.")
    
    print("✅ SUCCESS!")
    print(f"🤖 Gemini Response: {response.text}")
    print("\n🎉 Your XelBot is now fully functional!")
    
except Exception as e:
    print(f"❌ Error: {e}")
