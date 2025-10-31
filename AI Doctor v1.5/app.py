import os
import logging
import re
import json
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from flask import send_file
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Load environment variables FIRST, before any other operations
load_dotenv()

# Debug lines to confirm GOOGLE_API_KEY is found
print("=== API KEY DEBUG INFO ===")
print("GOOGLE_API_KEY from os.getenv:", os.getenv("GOOGLE_API_KEY"))
print("GOOGLE_API_KEY from os.environ:", os.environ.get("GOOGLE_API_KEY"))
print("Length of key from os.getenv:", len(os.getenv("GOOGLE_API_KEY") or ""))
print("Length of key from os.environ:", len(os.environ.get("GOOGLE_API_KEY") or ""))
print("Key starts with (os.getenv):", (os.getenv("GOOGLE_API_KEY") or "")[:10])
print("Key starts with (os.environ):", (os.environ.get("GOOGLE_API_KEY") or "")[:10])
print("=== END API KEY DEBUG INFO ===")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for Gemini availability and client
gemini_available = False
client = None
genai = None

# Import Google Generative AI with proper module access
def initialize_gemini():
    global gemini_available, client, genai
    
    try:
        # Try the standard import first
        import google.generativeai as genai_module
        genai = genai_module
        
        print("=== GEMINI INITIALIZATION DEBUG ===")
        
        # Configure the client with API key from environment
        print("Attempting to configure Google GenAI client (auto-detecting API key)...")
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Validate API key format
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        if len(api_key) < 30:
            raise ValueError("GOOGLE_API_KEY appears to be invalid (too short)")
            
        # Initialize the client
        # Use getattr to avoid linter issues
        configure_func = getattr(genai, 'configure')
        configure_func(api_key=api_key)
        client = genai
        print("Google GenAI client configured successfully")
        logger.info("Google GenAI client configured successfully")
        
        # Test the client with a simple request
        print("Testing client accessibility...")
        # Use getattr to avoid linter issues
        GenerativeModel = getattr(genai, 'GenerativeModel')
        model = GenerativeModel('gemini-2.5-flash')
        generate_content = getattr(model, 'generate_content')
        test_response = generate_content("Explain how AI works in a few words")
        print(f"Test response: {test_response.text}")
        print("Client initialized successfully")
        logger.info("Client initialized successfully")
        gemini_available = True

    except ImportError as e:
        # If the standard import fails, log the error
        print(f"ERROR: Google GenAI import error: {str(e)}")
        logger.error(f"Google GenAI import error: {str(e)}")
        gemini_available = False
        
    except Exception as e:
        print(f"ERROR: Google GenAI client configuration error: {str(e)}")
        logger.error(f"Google GenAI client configuration error: {str(e)}")
        gemini_available = False

# Initialize Gemini when the module loads
initialize_gemini()

print(f"=== END GEMINI INITIALIZATION DEBUG (gemini_available: {gemini_available}) ===")

# Language codes mapping
LANGUAGE_CODES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi'
}

# Simple translation function using a free API
def translate_text(text, target_language):
    """Translate text to target language using MyMemory API with maximum limits"""
    try:
        if target_language == 'en' or not text:
            return text
            
        # URL encode the text
        encoded_text = urllib.parse.quote(text)
        
        # Use MyMemory translation API with increased limits
        # Adding email parameter to increase quota (optional but helpful)
        # Using more segments for better translation quality
        url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair=en|{target_language}&de=example@example.com"
        
        # Make the request with increased timeout
        response = urllib.request.urlopen(url, timeout=30)
        data = json.loads(response.read())
        
        # Extract translated text
        if data.get('responseStatus') == 200 and data.get('responseData'):
            translated_text = data['responseData'].get('translatedText', text)
            logger.info(f"Translated '{text[:100]}...' to {target_language}: '{translated_text[:100]}...'")
            return translated_text
        else:
            logger.error(f"Translation API error: {data}")
            # Try fallback translation with shorter text if too long
            if len(text) > 500:
                # Split text into smaller chunks and translate each
                chunk_size = 400
                translated_chunks = []
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    chunk_encoded = urllib.parse.quote(chunk)
                    chunk_url = f"https://api.mymemory.translated.net/get?q={chunk_encoded}&langpair=en|{target_language}&de=example@example.com"
                    try:
                        chunk_response = urllib.request.urlopen(chunk_url, timeout=30)
                        chunk_data = json.loads(chunk_response.read())
                        if chunk_data.get('responseStatus') == 200 and chunk_data.get('responseData'):
                            translated_chunks.append(chunk_data['responseData'].get('translatedText', chunk))
                        else:
                            translated_chunks.append(chunk)
                    except Exception as chunk_error:
                        logger.error(f"Translation chunk error: {str(chunk_error)}")
                        translated_chunks.append(chunk)
                return ''.join(translated_chunks)
            return text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        # Return original text if translation fails
        return text

# Load system prompt
def load_system_prompt():
    try:
        with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback prompt
        return """You are Dr. Vaani, a friendly AI medical assistant. Provide helpful, direct answers to medical questions.
        
User concern: {user_message}
        
Dr. Vaani:"""

# Create Flask app - updated to serve React frontend
app = Flask(__name__, static_folder='frontend/dist')
# Updated CORS configuration to allow requests from Vercel deployment
CORS(app, origins=[
    "http://localhost:8080", 
    "http://127.0.0.1:8080", 
    "http://192.168.0.106:8080",
    "https://ai-doctor-blueplanet.vercel.app"
], supports_credentials=True)

# Load the system prompt
SYSTEM_PROMPT = load_system_prompt()

# Medical keywords to detect when to switch to medical mode
MEDICAL_KEYWORDS = [
    # English terms
    'fever', 'pain', 'headache', 'head ache', 'cough', 'cold', 'flu', 'vomit', 'nausea', 'stomach',
    'belly', 'diarrhea', 'constipation', 'rash', 'allergy', 'infection', 'injury',
    'wound', 'cut', 'burn', 'ache', 'sore', 'swelling', 'inflammation', 'blood',
    'pressure', 'diabetes', 'asthma', 'breathing', 'chest', 'heart', 'back',
    'joint', 'muscle', 'bone', 'fracture', 'dizziness', 'fatigue', 'tired',
    'sleep', 'appetite', 'weight', 'diet', 'nutrition', 'medicine', 'medication',
    'drug', 'tablet', 'pill', 'doctor', 'hospital', 'clinic', 'symptom', 'illness',
    'disease', 'condition', 'treatment', 'therapy', 'surgery', 'operation',
    'health', 'wellness', 'immunity', 'vaccine', 'immunization', 'vaccination',
    'temperature', 'pulse', 'heartbeat', 'sugar', 'glucose', 'cholesterol',
    'blood pressure', 'hypertension', 'hypotension', 'anxiety', 'depression',
    'stress', 'mental', 'psychological', 'mood', 'digestion', 'indigestion',
    'acid', 'reflux', 'heartburn', 'constipation', 'loose motion', 'motion',
    'urine', 'urination', 'kidney', 'liver', 'lungs', 'lung', 'throat', 'eye',
    'ear', 'nose', 'skin', 'hair', 'nail', 'tooth', 'teeth', 'gum', 'mouth',
    'pregnancy', 'pregnant', 'menstruation', 'period', 'cycle', 'menopause',
    'child', 'baby', 'infant', 'toddler', 'elderly', 'old age', 'senior',
    'immunocompromised', 'autoimmune', 'chronic', 'acute', 'malignant',
    'benign', 'cancer', 'tumor', 'growth', 'lesion', 'ulcer', 'infection',
    'viral', 'bacterial', 'fungal', 'parasitic', 'contagious', 'epidemic',
    'pandemic', 'outbreak', 'endemic', 'sporadic', 'incidence', 'prevalence',
    'mortality', 'morbidity', 'prognosis', 'diagnosis', 'diagnostic',
    'screening', 'test', 'examination', 'exam', 'checkup', 'physical',
    'preventive', 'prophylactic', 'prophylaxis', 'protection', 'prevent',
    'avoid', 'risk', 'factor', 'cause', 'etiology', 'pathophysiology',
    'pathology', 'histology', 'anatomy', 'physiology', 'biochemistry',
    'metabolism', 'metabolic', 'endocrine', 'hormone', 'enzyme', 'genetic',
    'hereditary', 'congenital', 'acquired', 'idiopathic', 'iatrogenic',
    'complication', 'sequela', 'side effect', 'adverse', 'reaction', 'allergic',
    'anaphylaxis', 'shock', 'emergency', 'urgent', 'critical', 'life-threatening',
    'terminal', 'palliative', 'hospice', 'rehabilitation', 'recovery',
    'convalescence', 'healing', 'remission', 'relapse', 'recurrence',
    'prognosis', 'outcome', 'survival', 'quality of life', 'function',
    'disability', 'impairment', 'deficit', 'deficiency', 'insufficiency',
    'failure', 'insufficiency', 'overload', 'strain', 'stress', 'fatigue',
    
    # Hindi terms
    'sar dard', 'sardard', 'khansi', 'kharash', 'jvar', 'bukhar', 'dast', 'daura',
    'pet dard', 'pet mein dard', 'najla', 'nakli', 'nakli', 'takleef', 'taklif',
    'dava', 'dawai', 'gutkha', 'goli', 'tablet', 'chikitsa', 'upachar', 'ilaj',
    'sujan', 'sujak', 'daur', 'daur', 'daur', 'daur',
    
    # Marathi terms
    'pot dukhtay', 'potat dukhtay', 'pot dukhte', 'potat dukhte', 'potache dukhte',
    'potache dukhtay', 'potache dukh', 'potat dukh', 'pot dukh', 'mala pot dukhtoy',
    'mala potat dukhtoy', 'majha pot dukhtay', 'majha potat dukhtay', 'majha pot dukhte',
    'majha potat dukhte', 'sardar', 'khokhale', 'khokhlya', 'khokhalyat', 'khokhlyat',
    'bukhar', 'taklif', 'takleef', 'davakhana', 'dawai', 'gutkha', 'goli', 'tablet',
    'upachar', 'ilaj', 'sujan', 'sujak', 'daur', 'daura', 'jvar', 'nakli', 'naklya',
    'doka', 'dokyache dukh', 'doka dukhtay', 'dokaat dukhtay', 'mala dokaat dukhtoy',
    'majha doka dukhtay', 'majha dokaat dukhtay', 'dokyaat dukhtay', 'dokyaat dukhte'
]

# Simple greetings that should not trigger medical responses
SIMPLE_GREETINGS = [
    'hi', 'hello', 'hey', 'hii', 'helo', 'hai', 'hy', 'hyy', 'hiii', 'heyy',
    'good morning', 'good afternoon', 'good evening', 'gm', 'ga', 'ge',
    'namaste', 'namaskar', 'vanakkam', 'namaskaram'
]

def is_medical_query(message):
    """Check if the message contains medical keywords"""
    message_lower = message.lower().strip()
    
    # Check if it's a simple greeting
    if message_lower in SIMPLE_GREETINGS:
        return False
    
    # Check for medical keywords
    if any(keyword in message_lower for keyword in MEDICAL_KEYWORDS):
        return True
    
    # Additional check for common medical phrases
    medical_phrases = [
        'dard', 'dard ho', 'dard hai', 'bimar', 'bimari', 'takleef', 'taklif',
        'problem', 'pain in', 'ache in', 'hurts', 'kr rha', 'kr rha h', 'lag rha',
        'dukhtay', 'dukhte', 'dukh', 'pot', 'sardar', 'khansi', 'bukhar', 'jvar',
        'doka', 'dokya', 'dokaat', 'dokyache'
    ]
    
    return any(phrase in message_lower for phrase in medical_phrases)

def is_simple_greeting(message):
    """Check if the message is a simple greeting"""
    message_lower = message.lower().strip()
    # Check for exact matches and simple greetings
    if message_lower in SIMPLE_GREETINGS:
        return True
    
    # Check if it's primarily a greeting (more than 50% greeting words)
    words = message_lower.split()
    if len(words) == 0:
        return False
    
    greeting_words = sum(1 for word in words if word in SIMPLE_GREETINGS)
    return (greeting_words / len(words)) > 0.5

def clean_response(response_text):
    """Clean response for direct presentation"""
    # Remove extra whitespace but preserve paragraph breaks
    cleaned = re.sub(r'\n\s*\n', '\n\n', response_text).strip()
    # Remove all asterisks (bold, italic, bullet points)
    cleaned = re.sub(r'\*', '', cleaned)
    # Remove hash symbols
    cleaned = re.sub(r'#', '', cleaned)
    # Remove other common markdown formatting
    cleaned = re.sub(r'__([^_]+)__', r'\1', cleaned)  # Remove underline
    cleaned = re.sub(r'~~([^~]+)~~', r'\1', cleaned)  # Remove strikethrough
    # Convert markdown bullet points to plain hyphens
    cleaned = re.sub(r'^\s*\*\s', '- ', cleaned, flags=re.MULTILINE)
    # Convert numbered lists to bullet points
    cleaned = re.sub(r'^\s*\d+\.\s', '- ', cleaned, flags=re.MULTILINE)
    # Ensure proper spacing after bullet points
    cleaned = re.sub(r'-([^\s])', r'- \1', cleaned)
    # Remove extra spaces
    cleaned = re.sub(r' +', ' ', cleaned)
    # Remove leading and trailing spaces on each line
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    # Remove any remaining standalone formatting characters
    cleaned = re.sub(r'^\s*[*#]+\s*$', '', cleaned, flags=re.MULTILINE)
    # Ensure complete sentences in bullet points
    cleaned = re.sub(r'-\s*\n', '- ', cleaned)
    return cleaned

@app.route('/')
def home():
    # Serve the React frontend index.html
    try:
        return send_from_directory('frontend/dist', 'index.html')
    except:
        return "Frontend build not found. Please run 'npm run build' in the frontend directory."

# Specific routes for static assets with proper MIME types
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    try:
        return send_from_directory('frontend/dist/assets', filename)
    except:
        return jsonify({"error": "Asset not found"}), 404

# Add more specific routes for common file types
@app.route('/<filename>.js')
def serve_js(filename):
    try:
        return send_from_directory('frontend/dist', f'{filename}.js')
    except:
        # For client-side routing, serve index.html for non-API routes
        if not filename.startswith(('chat', 'translate', 'preprocess', 'health')):
            try:
                return send_from_directory('frontend/dist', 'index.html')
            except:
                return "Frontend build not found. Please run 'npm run build' in the frontend directory."
        else:
            return jsonify({"error": "Not found"}), 404

@app.route('/<filename>.css')
def serve_css(filename):
    try:
        return send_from_directory('frontend/dist', f'{filename}.css')
    except:
        return send_from_directory('frontend/dist', 'index.html')

@app.route('/<filename>.ico')
def serve_ico(filename):
    try:
        return send_from_directory('frontend/dist', f'{filename}.ico')
    except:
        return send_from_directory('frontend/dist', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get user message and language from request
        data = request.get_json()
        user_message = data.get('message', '') if data else ''
        target_language = data.get('language', 'en') if data else 'en'
        translate_to = data.get('translate_to', 'en') if data else 'en'
        source_language = data.get('source_language', 'en') if data else 'en'
        
        # Log the translation parameters for debugging
        logger.info(f"Received translation parameters - target_language: {target_language}, translate_to: {translate_to}, source_language: {source_language}")
        
        # For dynamic translation, we want to translate the AI response to the selected language
        # translate_to is the language we want to translate the response to
        response_language = translate_to
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Store the original message for AI processing
        original_message = user_message
        
        # Preprocess the message to English for better medical query detection
        processed_message = user_message
        if source_language != 'en':
            try:
                # Translate to English for medical keyword detection
                processed_message = translate_text(user_message, 'en')
                logger.info(f"Preprocessed message from {source_language} to English: '{user_message}' -> '{processed_message}'")
            except Exception as preprocess_error:
                logger.error(f"Preprocessing error: {str(preprocess_error)}")
                # Continue with original message if preprocessing fails
                processed_message = user_message
        
        # Check if this is a simple greeting
        if is_simple_greeting(processed_message):
            greeting_response = "Hello! I'm Dr. Vaani, your friendly AI health assistant. I'm here to help you with any health concerns or just have a chat. What would you like to discuss today? 😊"
            # Translate to target language if needed
            if response_language != 'en':
                translated_response = translate_text(greeting_response, response_language)
                logger.info(f"Translated greeting to {response_language}: {translated_response}")
                return Response(translated_response, mimetype='text/plain')
            return Response(greeting_response, mimetype='text/plain')
        
        # Check if this is a medical query using the preprocessed (English) message
        is_medical = is_medical_query(processed_message)
        
        # Use the system prompt
        system_prompt = SYSTEM_PROMPT
        
        # If Gemini is not available, return a more detailed error response
        if not gemini_available or client is None:
            error_response = "AI service temporarily unavailable. "
            # Add more specific error information
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                error_response += "GOOGLE_API_KEY not found in environment variables. "
            elif api_key == "":
                error_response += "GOOGLE_API_KEY is empty. "
            else:
                error_response += "Please check your API key configuration. "
            error_response += "Contact administrator for assistance."
            
            logger.error("Gemini model is not available for requests - check API key configuration")
            # Translate to target language if needed
            if response_language != 'en':
                translated_error = translate_text(error_response, response_language)
                logger.info(f"Translated error to {response_language}: {translated_error}")
                return Response(translated_error, mimetype='text/plain')
            return Response(error_response, mimetype='text/plain')
        
        # Prepare the prompt with system context
        # Use the ORIGINAL message for the AI to maintain context in the user's language
        # But use the processed message for determining if it's medical
        prompt = system_prompt.format(user_message=original_message)
        
        # Add instruction about mode based on query type for a full-fledged AI bot
        if is_medical:
            prompt += "\n\nCRITICAL INSTRUCTIONS: User has medical concerns. You MUST follow the EXACT structured format from the system prompt with these specific requirements:"
            prompt += "\n1. Treatment (5 points) - exactly 5 bullet points"
            prompt += "\n2. Precautions (3 points) - exactly 3 bullet points" 
            prompt += "\n3. Reasons (3 points) - exactly 3 bullet points"
            prompt += "\n4. When to see doctor (2 points) - exactly 2 bullet points"
            prompt += "\nUse the EXACT section titles as shown in the system prompt example."
            prompt += "\nUse bullet points with hyphens as shown in the example."
            prompt += "\nKeep response under 20 lines total."
            # Add specific instruction to respond in the user's language
            prompt += f"\n\nIMPORTANT: Respond in the same language as the user's message: {original_message}"
        else:
            prompt += "\n\nIMPORTANT: User wants general conversation. Respond naturally and briefly. Be helpful but don't provide medical advice unless explicitly asked."
            # Add specific instruction to respond in the user's language
            prompt += f"\n\nIMPORTANT: Respond in the same language as the user's message: {original_message}"
        
        logger.info(f"Sending prompt to Gemini: {prompt[:100]}...")  # Log first 100 chars
        logger.info(f"Original message: '{user_message}', Processed message: '{processed_message}', Is medical: {is_medical}")
        
        # Generate response using Gemini with streaming
        try:
            # Use the global genai module with getattr to avoid linter issues
            if genai is not None:
                GenerativeModel = getattr(genai, 'GenerativeModel')
                model = GenerativeModel('gemini-2.5-flash')
                generate_content = getattr(model, 'generate_content')
                response = generate_content(prompt, stream=True)
            else:
                raise Exception("GenAI module not available")
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            error_msg = f"AI service error: {str(e)}"
            if response_language != 'en':
                error_msg = translate_text(error_msg, response_language)
                logger.info(f"Translated error to {response_language}: {error_msg}")
            return Response(error_msg, mimetype='text/plain')
        
        def generate():
            full_response = ""
            try:
                for chunk in response:
                    if chunk.text:
                        # Clean each chunk before yielding
                        cleaned_chunk = clean_response(chunk.text)
                        full_response += chunk.text  # Keep original for logging
                        # IMPORTANT: Do NOT translate the response chunks as this may disrupt the structured format
                        # The AI should generate the response in the correct language directly
                        yield cleaned_chunk
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                # If streaming fails, provide a simple error response
                error_msg = "I'm having trouble responding right now. Please try again."
                if response_language != 'en':
                    error_msg = translate_text(error_msg, response_language)
                    logger.info(f"Translated error to {response_language}: {error_msg}")
                yield error_msg
            
            # Log the complete response
            logger.info(f"Complete response: {full_response[:100]}...")
        
        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        error_msg = f"Service error: {str(e)}. Please try again later."
        target_language = request.get_json().get('language', 'en') if request.get_json() else 'en'
        translate_to = request.get_json().get('translate_to', 'en') if request.get_json() else 'en'
        response_language = translate_to
        if response_language != 'en':
            error_msg = translate_text(error_msg, response_language)
            logger.info(f"Translated error to {response_language}: {error_msg}")
        return Response(error_msg, mimetype='text/plain')

# Endpoint for translating text to a target language
@app.route('/translate', methods=['POST'])
def translate_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('target_language', 'en')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
            
        translated_text = translate_text(text, target_language)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({"error": f"Translation service error: {str(e)}"}), 500

# Endpoint for translating user input to English for medical query processing
@app.route('/preprocess', methods=['POST'])
def preprocess_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        source_language = data.get('source_language', 'en')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
            
        # Translate to English for medical keyword detection
        if source_language != 'en':
            translated_text = translate_text(text, 'en')
            return jsonify({
                "original_text": text,
                "processed_text": translated_text,
                "source_language": source_language
            })
        else:
            return jsonify({
                "original_text": text,
                "processed_text": text,
                "source_language": source_language
            })
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return jsonify({"error": f"Preprocessing service error: {str(e)}"}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint to verify environment variables"""
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    return {
        "status": "ok",
        "gemini_available": gemini_available,
        "api_key_exists": api_key is not None,
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_preview": api_key[:10] if api_key else None
    }

# Catch-all route for client-side routing (must be last)
@app.route('/<path:path>')
def serve_static(path):
    # For client-side routing, serve index.html for all non-API routes
    if not path.startswith(('api/', 'chat', 'translate', 'preprocess', 'health')):
        try:
            return send_from_directory('frontend/dist', 'index.html')
        except:
            return "Frontend build not found. Please run 'npm run build' in the frontend directory."
    else:
        return jsonify({"error": "Not found"}), 404

# Vercel requires the app to be exported as `app`
# Keep the if __name__ == '__main__' block for local testing
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

app = app