import google.generativeai as genai

# List available models
API_KEY = "AIzaSyA0ePTaWI-Up6oTuG_R_B-Pvave0UwumjM"

try:
    genai.configure(api_key=API_KEY)
    print("🔍 Available Gemini Models:")
    print("=" * 40)
    
    models = list(genai.list_models())
    for model in models:
        print(f"📋 {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            methods = list(model.supported_generation_methods)
            if 'generateContent' in methods:
                print(f"   ✅ Supports generateContent")
            else:
                print(f"   ❌ Methods: {methods}")
        print()
        
except Exception as e:
    print(f"❌ Error: {e}")
