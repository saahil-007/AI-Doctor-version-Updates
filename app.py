import os
import logging
import re
import json
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, Response, send_from_directory  # pyright: ignore[reportMissingImports]
from flask_cors import CORS  # pyright: ignore[reportMissingModuleSource]
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from collections import deque

# Load environment variables FIRST, before any other operations
load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for Gemini availability and client
gemini_available = False
client = None
genai = None
gemini_model = None  # Single instance of the Gemini model

# Context window to store chat history (10 chats)
chat_context = {}

# Required variables for medical consultation
REQUIRED_VARIABLES = [
    "age",
    "gender", 
    "symptoms_duration",
    "recent_medical_history",
    "allergies",
    "chronic_diseases",
    "symptom_specifications",
    "more_details"
]

# Variable questions to ask the user

# Import Google Generative AI with proper module access
def initialize_gemini():
    global gemini_available, client, genai, gemini_model
    
    try:
        # Try the standard import first
        import google.generativeai as genai_module  # pyright: ignore[reportMissingImports]
        genai = genai_module
        
        print("=== GEMINI INITIALIZATION DEBUG ===")
        
        # Configure the client with API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Print the API key info for debugging (without revealing the full key)
        print(f"GOOGLE_API_KEY from os.getenv: {'FOUND' if api_key else 'NOT FOUND'}")
        
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
        
        # Create a single instance of the Gemini model for reuse
        GenerativeModel = getattr(genai, 'GenerativeModel')
        gemini_model = GenerativeModel('gemini-2.5-flash')
        print("Google GenAI model instance created successfully")
        logger.info("Google GenAI model instance created successfully")
        
        # Test the client with a simple request
        print("Testing client accessibility...")
        generate_content = getattr(gemini_model, 'generate_content')
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
            # Remove the text length limit - translate all texts regardless of length
            # Split text into smaller chunks and translate each
            # For word-to-word translation, split by sentences
            import re
            sentences = re.split(r'[.!?]+', text)
            translated_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence_encoded = urllib.parse.quote(sentence)
                    sentence_url = f"https://api.mymemory.translated.net/get?q={sentence_encoded}&langpair=en|{target_language}&de=example@example.com"
                    try:
                        sentence_response = urllib.request.urlopen(sentence_url, timeout=30)
                        sentence_data = json.loads(sentence_response.read())
                        if sentence_data.get('responseStatus') == 200 and sentence_data.get('responseData'):
                            translated_sentences.append(sentence_data['responseData'].get('translatedText', sentence))
                        else:
                            translated_sentences.append(sentence)
                    except Exception as sentence_error:
                        logger.error(f"Translation sentence error: {str(sentence_error)}")
                        translated_sentences.append(sentence)
            return ' '.join(translated_sentences)
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

def extract_variables_with_gemini(user_id, message):
    """Use Gemini to extract all variables from user message and chat history with high accuracy"""
    global genai, gemini_model
    
    # Use Gemini to extract variables
    try:
        # Build context from chat history
        context_parts = []
        if user_id in chat_context and chat_context[user_id]["chat_history"]:
            # Get last few messages for context
            recent_messages = list(chat_context[user_id]["chat_history"])[-5:]
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Dr. Vaani"
                context_parts.append(f"{role}: {msg['message']}")
        
        context = "\n".join(context_parts)
        
        # Create prompt for Gemini to extract variables
        prompt = f"""
You are an expert medical information extractor and conversational assistant. Your task is to analyze the user's message and extract ALL specific medical variables with extreme precision from a SINGLE message.

Context (conversation history):
{context}

Current user message: {message}

Extract the following variables if present in the message or context:
1. age: Patient's age in years (number only) - Look for phrases like "my X year old son/daughter", "X years old", "I'm 20 years old", "20 year old", "20 yrs", etc.
2. gender: Patient's gender (Male/Female/Other) - Look for words like "male", "female", "man", "woman", "boy", "girl", "son" (implies Male), or "daughter" (implies Female)
3. symptoms_duration: How long the patient has been experiencing symptoms (be very specific, e.g., "since yesterday", "3 days", "since this morning")
4. recent_medical_history: Any recent medical history related to symptoms
5. allergies: Any allergies related to the symptoms
6. chronic_diseases: Any chronic diseases related to the symptoms
7. symptom_specifications: Detailed specifications about the symptoms
8. more_details: Any additional relevant information

CRITICAL INSTRUCTIONS:
- Extract ALL information from the SINGLE current message - do not wait for multiple messages
- For a message like "im 20 , male", extract BOTH age=20 AND gender=Male
- For a message like "my 5 year old son has fever", extract age=5 AND gender=Male
- For a message like "hello my 7 year daughter is having running nose", extract age=7 AND gender=Female (the patient is the child, not the parent)
- Extract information with extreme precision and accuracy
- When someone says "my son" or "my daughter", they are referring to their child as the patient, not themselves
- The person seeking medical advice may be a parent or guardian asking about their child's symptoms
- If a variable is not clearly stated, do not guess - leave it out
- Respond ONLY in JSON format with the extracted variables
- If no variables can be extracted, return an empty JSON object {{}}
- Additionally, if you need to ask the user for more information, suggest a natural, conversational question that would help collect the missing information

Example response formats:
For message "im 20 , male":
{{
  "age": "20",
  "gender": "Male"
}}

For message "my 5 year old son has fever":
{{
  "age": "5",
  "gender": "Male",
  "symptom_specifications": "fever"
}}

For message "hello my 7 year daughter is having running nose":
{{
  "age": "7",
  "gender": "Female",
  "symptom_specifications": "running nose"
}}

For message "I have headache since yesterday":
{{
  "symptoms_duration": "since yesterday",
  "symptom_specifications": "headache"
}}

JSON Response:"""
        
        # Use the single Gemini model instance
        if gemini_model is not None:
            generate_content = getattr(gemini_model, 'generate_content')
            response = generate_content(prompt)
        else:
            raise Exception("Gemini model not available")
        
        # Parse the JSON response
        import json
        try:
            extracted = json.loads(response.text.strip())
            return extracted
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty dict
            logger.warning("Gemini variable extraction failed")
            return {}
            
    except Exception as e:
        logger.error(f"Error in Gemini variable extraction: {str(e)}")
        # Return empty dict if Gemini fails
        return {}


def generate_variable_question_with_gemini(user_id, variable_name, user_message):
    """Use Gemini to generate a contextual question for a specific variable"""
    global genai, gemini_model
    
    # Use Gemini to generate the question
    try:
        # Build context from chat history
        context_parts = []
        if user_id in chat_context and chat_context[user_id]["chat_history"]:
            # Get last few messages for context
            recent_messages = list(chat_context[user_id]["chat_history"])[-5:]
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Dr. Vaani"
                context_parts.append(f"{role}: {msg['message']}")
        
        context = "\n".join(context_parts)
        
        # Create prompt for Gemini to generate a question
        prompt = f"""
You are an expert medical conversational assistant. Your task is to generate a natural, contextual question to collect a specific piece of medical information from the user.

The variable you need to ask about is: {variable_name}

Context (conversation history):
{context}

Current user message: {user_message}

Generate a natural, conversational question to collect information about {variable_name}. 
The question should be:
1. Contextually relevant based on the conversation history
2. Natural and conversational, not robotic
3. Specific to the variable being collected
4. Easy for the user to understand and answer

Example outputs:
- For age: "Could you share your age with me?"
- For gender: "May I know your gender?"
- For symptoms_duration: "How long have you been experiencing these symptoms?"

Question:"""
        
        # Use the single Gemini model instance
        if gemini_model is not None:
            generate_content = getattr(gemini_model, 'generate_content')
            response = generate_content(prompt)
        else:
            raise Exception("Gemini model not available")
        
        # Return the generated question
        question = response.text.strip()
        return question if question else f"Please provide information about {variable_name.replace('_', ' ')}."
            
    except Exception as e:
        logger.error(f"Error in Gemini question generation: {str(e)}")
        # Return a generic question if Gemini fails
        return f"Could you please provide information about your {variable_name.replace('_', ' ')}?"

# Context window functions for variable collection
def initialize_user_context(user_id):
    """Initialize context for a new user"""
    chat_context[user_id] = {
        "chat_history": deque(maxlen=10),  # Keep only last 10 messages
        "collected_variables": {},
        "current_variable": None,
        "variables_collected": False
    }

def add_to_chat_history(user_id, role, message):
    """Add a message to the user's chat history"""
    if user_id not in chat_context:
        initialize_user_context(user_id)
    
    chat_context[user_id]["chat_history"].append({
        "role": role,
        "message": message
    })

def get_next_uncollected_variable(user_id):
    """Get the next variable that needs to be collected"""
    if user_id not in chat_context:
        initialize_user_context(user_id)
    
    collected = chat_context[user_id]["collected_variables"]
    
    # Prioritize required variables
    required_vars = ["age", "gender", "symptoms_duration"]
    for variable in required_vars:
        if variable not in collected or not collected[variable]:
            return variable
    
    # Then check other variables
    for variable in REQUIRED_VARIABLES:
        if variable not in collected or not collected[variable]:
            return variable
    
    return None

def collect_variable(user_id, variable, value):
    """Collect a variable value from the user"""
    if user_id not in chat_context:
        initialize_user_context(user_id)
    
    # Skip optional variables if user says "No" or similar
    if variable in ["recent_medical_history", "allergies", "chronic_diseases"] and value.lower() in ["no", "nope", "none", "nothing", "n/a", "na", "n"]:
        value = "None reported"
    
    chat_context[user_id]["collected_variables"][variable] = value
    
    # Check if all required variables are collected
    next_var = get_next_uncollected_variable(user_id)
    if next_var is None:
        chat_context[user_id]["variables_collected"] = True
    
    return next_var

def get_user_context_prompt(user_id):
    """Generate a prompt with the collected user context"""
    if user_id not in chat_context or not chat_context[user_id]["variables_collected"]:
        return ""
    
    variables = chat_context[user_id]["collected_variables"]
    context_parts = []
    
    for variable in REQUIRED_VARIABLES:
        if variable in variables and variables[variable]:
            # Format variable name for readability
            formatted_name = variable.replace("_", " ").title()
            context_parts.append(f"{formatted_name}: {variables[variable]}")
    
    if context_parts:
        return "\n\nPatient Information:\n" + "\n".join(context_parts)
    
    return ""

def reset_user_context(user_id):
    """Reset the user context after providing a medical response"""
    if user_id in chat_context:
        chat_context[user_id]["collected_variables"] = {}
        chat_context[user_id]["current_variable"] = None
        chat_context[user_id]["variables_collected"] = False

def provide_medical_response(user_id, original_message, response_language, source_language):
    """Provide the final structured medical response after collecting all variables"""
    global gemini_model
    
    # Use the system prompt
    system_prompt = SYSTEM_PROMPT
    
    # Prepare the prompt with system context
    # Use the ORIGINAL message for the AI to maintain context in the user's language
    prompt = system_prompt.format(user_message=original_message)
    
    # Add patient information context
    patient_context = get_user_context_prompt(user_id)
    if patient_context:
        prompt += patient_context
    
    # Extract symptoms from the context to suggest relevant medicines
    symptoms_info = ""
    if user_id in chat_context and "collected_variables" in chat_context[user_id]:
        variables = chat_context[user_id]["collected_variables"]
        if "symptom_specifications" in variables and variables["symptom_specifications"]:
            symptoms_info = variables["symptom_specifications"]
        elif "more_details" in variables and variables["more_details"]:
            symptoms_info = variables["more_details"]
    
    # Add medicine suggestions to the prompt if symptoms are available
    if symptoms_info:
        medicine_suggestions = get_contextual_medicine_suggestions(symptoms_info)
        if medicine_suggestions:
            prompt += f"\n\n{medicine_suggestions}"
    
    # Add instruction about mode based on query type for a full-fledged AI bot
    prompt += "\n\nCRITICAL INSTRUCTIONS: User has medical concerns. You MUST follow the EXACT structured format from the system prompt with these specific requirements:"
    prompt += "\n1. General Treatment (5 points) - exactly 5 bullet points"
    prompt += "\n2. Medical Treatment (5 points) - exactly 5 bullet points"
    prompt += "\n3. Precautions (3 points) - exactly 3 bullet points" 
    prompt += "\n4. Reasons (3 points) - exactly 3 bullet points"
    prompt += "\n5. When to see doctor (2 points) - exactly 2 bullet points"
    prompt += "\n6. Acknowledgement and Conclusion - exactly 1 section"
    prompt += "\nUse the EXACT section titles as shown in the system prompt example."
    prompt += "\nUse bullet points with hyphens as shown in the example."
    prompt += "\nKeep response under 30 lines total."
    # Add specific instruction to respond in the user's language
    prompt += f"\n\nIMPORTANT: Respond in the same language as the user's message: {original_message}"
    
    # Add specific instruction about suggesting Indian medicines with dosage when appropriate
    if symptoms_info:
        prompt += f"\n\nADDITIONAL INSTRUCTIONS: Based on the symptoms described ('{symptoms_info}'), when providing medical treatment recommendations in the 'Medical Treatment' section, provide exactly 5 treatment points including relevant Indian medicines with specific dosage instructions (e.g., 'take three times a day', 'take twice daily'). Include the medicine name, manufacturer, key composition, dosage frequency, and duration when suggesting medicines. ALWAYS explain WHY each medicine is being suggested for specific symptoms. Always emphasize that these are suggestions and a doctor's consultation is necessary for proper diagnosis and prescription.\n\nEXAMPLE FORMAT:\nMedical Treatment (5 points):\n- Take [Medicine Name] by [Manufacturer] ([Composition]) - [Dosage Instructions] for [Duration] for [specific symptom relief] - [Explanation of why this medicine helps with these symptoms]\n- Take [Medicine Name] by [Manufacturer] ([Composition]) - [Dosage Instructions] for [Duration] for [specific symptom relief] - [Explanation of why this medicine helps with these symptoms]\n(Repeat for exactly 5 points)"
    
    logger.info(f"Sending prompt to Gemini: {prompt[:100]}...")  # Log first 100 chars
    
    # Generate response using Gemini (non-streaming for better language detection)
    try:
        # Use the single Gemini model instance
        if gemini_model is not None:
            generate_content = getattr(gemini_model, 'generate_content')
            response = generate_content(prompt)  # Non-streaming for full response
        else:
            raise Exception("Gemini model not available")
    except Exception as e:
        logger.error(f"Error generating content with Gemini: {str(e)}")
        # Return error response when Gemini fails
        error_response = "I'm currently experiencing technical difficulties. Please try again later."
        # Add AI response to chat history
        add_to_chat_history(user_id, "assistant", error_response)
        # Reset user context
        reset_user_context(user_id)
        return Response(error_response, mimetype='text/plain')
    
    # Get the full response text
    full_response = response.text if hasattr(response, 'text') else str(response)
    
    # Clean the response
    cleaned_response = clean_response(full_response)
    
    # Add AI response to chat history
    add_to_chat_history(user_id, "assistant", cleaned_response)
    
    # Log the complete response
    logger.info(f"Complete response: {cleaned_response[:100]}...")
    
    # Reset user context for next consultation
    reset_user_context(user_id)
    
    # Translate the entire response if needed - no limits on translation
    if response_language != 'en':
        translated_response = translate_text(cleaned_response, response_language)
        logger.info(f"Translated response to {response_language}: {translated_response[:100]}...")
        return Response(translated_response, mimetype='text/plain')
    
    return Response(cleaned_response, mimetype='text/plain')

# Create Flask app - updated to serve React frontend
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Increase to 50MB limit for larger requests

# Updated CORS configuration to allow requests from Vercel deployment
# Allow all origins to simplify deployment across Vercel preview and production URLs
CORS(app, origins="*", supports_credentials=False)

# Load system prompt
SYSTEM_PROMPT = load_system_prompt()

# Function to fetch data from local dataset files
def fetch_local_dataset_data():
    """
    Fetch medical data from local dataset files.
    """
    try:
        logger.info("Fetching medical data from local dataset files")
        
        # Load medical keywords
        with open('data/medical_keywords.json', 'r', encoding='utf-8') as f:
            keywords_data = json.load(f)
            # Flatten all keywords from all categories into a single list
            all_keywords = []
            for category, keywords in keywords_data['medical_keywords'].items():
                all_keywords.extend(keywords)
            medical_keywords = all_keywords
            
        # Load simple greetings
        with open('data/simple_greetings.json', 'r', encoding='utf-8') as f:
            greetings_data = json.load(f)
            simple_greetings = greetings_data['simple_greetings']
            
        # Load medical phrases
        with open('data/medical_phrases.json', 'r', encoding='utf-8') as f:
            phrases_data = json.load(f)
            medical_phrases = phrases_data['medical_phrases']
            
        # Load Indian medicines data
        with open('data/indian_medicines.json', 'r', encoding='utf-8') as f:
            indian_medicines = json.load(f)
            
        return {
            'medical_keywords': medical_keywords,
            'simple_greetings': simple_greetings,
            'medical_phrases': medical_phrases,
            'indian_medicines': indian_medicines
        }
    except Exception as e:
        logger.error(f"Error fetching local dataset data: {str(e)}")
        # Fallback to empty lists
        return {
            'medical_keywords': [],
            'simple_greetings': [],
            'medical_phrases': [],
            'indian_medicines': []
        }

# Load medical data (from local files)
medical_data = fetch_local_dataset_data()
MEDICAL_KEYWORDS = medical_data['medical_keywords']
SIMPLE_GREETINGS = medical_data['simple_greetings']
MEDICAL_PHRASES = medical_data['medical_phrases']
INDIAN_MEDICINES = medical_data['indian_medicines']

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
    return any(phrase in message_lower for phrase in MEDICAL_PHRASES)

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
        user_id = data.get('user_id', 'default_user') if data else 'default_user'
        
        # Log the translation parameters for debugging
        logger.info(f"Received translation parameters - target_language: {target_language}, translate_to: {translate_to}, source_language: {source_language}")
        
        # For dynamic translation, we want to translate the AI response to the selected language
        # translate_to is the language we want to translate the response to
        response_language = translate_to
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Store the original message for AI processing
        original_message = user_message
        
        # Add user message to chat history
        add_to_chat_history(user_id, "user", user_message)
        
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
            greeting_response = "Hello! I'm Dr. Vaani, your caring AI health assistant. I'm here to help you with any health concerns with a gentle, nurturing approach. What would you like to discuss today? 😊"
            # Add AI response to chat history
            add_to_chat_history(user_id, "assistant", greeting_response)
            # Translate to target language if needed
            if response_language != 'en':
                translated_response = translate_text(greeting_response, response_language)
                logger.info(f"Translated greeting to {response_language}: {translated_response}")
                return Response(translated_response, mimetype='text/plain')
            return Response(greeting_response, mimetype='text/plain')
        
        # Check if this is a medical query using the preprocessed (English) message
        is_medical = is_medical_query(processed_message)
        
        # If we're already in the middle of collecting medical variables, 
        # continue with medical consultation regardless of current message content
        if user_id in chat_context and not chat_context[user_id].get("variables_collected", False):
            is_medical = True
        
        # If not a medical query, provide general response
        if not is_medical:
            # Use the system prompt
            system_prompt = SYSTEM_PROMPT
            
            # Prepare the prompt with system context
            prompt = system_prompt.format(user_message=original_message)
            
            prompt += "\n\nIMPORTANT: User wants general conversation. Respond naturally and briefly. Be helpful but don't provide medical advice unless explicitly asked."
            # Add specific instruction to respond in the user's language
            prompt += f"\n\nIMPORTANT: Respond in the same language as the user's message: {original_message}"
            
            logger.info(f"Sending prompt to Gemini: {prompt[:100]}...")  # Log first 100 chars
            
            # Generate response using Gemini (non-streaming for better language detection)
            try:
                # Use the single Gemini model instance
                if gemini_model is not None:
                    response = gemini_model.generate_content(prompt)
                else:
                    raise Exception("Gemini model not available")
            except Exception as e:
                logger.error(f"Error generating content with Gemini: {str(e)}")
                # Return error response when Gemini fails
                error_response = "I'm currently experiencing technical difficulties. Please try again later."
                if response_language != 'en':
                    error_response = translate_text(error_response, response_language)
                    logger.info(f"Translated error response to {response_language}: {error_response}")
                # Add AI response to chat history
                add_to_chat_history(user_id, "assistant", error_response)
                return Response(error_response, mimetype='text/plain')
            
            # Get the full response text
            full_response = response.text if hasattr(response, 'text') else str(response)
            
            # Clean the response
            cleaned_response = clean_response(full_response)
            
            # Add AI response to chat history
            add_to_chat_history(user_id, "assistant", cleaned_response)
            
            # Log the complete response
            logger.info(f"Complete response: {cleaned_response[:100]}...")
            
            # Translate the entire response if needed - no limits on translation
            if response_language != 'en':
                translated_response = translate_text(cleaned_response, response_language)
                logger.info(f"Translated response to {response_language}: {translated_response[:100]}...")
                return Response(translated_response, mimetype='text/plain')
            
            return Response(cleaned_response, mimetype='text/plain')
        
        # Handle medical query with streamlined variable collection workflow
        # Check if we're in the middle of collecting variables
        if user_id in chat_context and not chat_context[user_id].get("variables_collected", False):
            # If we're collecting a variable, store the user's response
            current_variable = chat_context[user_id].get("current_variable")
            if current_variable:
                # Collect the variable value
                next_variable = collect_variable(user_id, current_variable, user_message)
                
                # If all variables are collected, provide the medical response
                if next_variable is None:
                    # All variables collected, provide medical response
                    return provide_medical_response(user_id, original_message, response_language, source_language)
                else:
                    # Check if we already have this information
                    if next_variable in chat_context[user_id]["collected_variables"] and chat_context[user_id]["collected_variables"][next_variable]:
                        # Skip this variable and get the next one
                        next_variable = get_next_uncollected_variable(user_id)
                        if next_variable is None:
                            # All variables collected, provide medical response
                            chat_context[user_id]["variables_collected"] = True
                            return provide_medical_response(user_id, original_message, response_language, source_language)
                    
                    # If we still need this variable, generate a question for it
                    question = generate_variable_question_with_gemini(user_id, next_variable, user_message)
                    chat_context[user_id]["current_variable"] = next_variable
                    # Add AI response to chat history
                    add_to_chat_history(user_id, "assistant", question)
                    return Response(question, mimetype='text/plain')
            else:
                # First message in conversation - extract all variables at once
                extracted_variables = extract_variables_with_gemini(user_id, user_message)
                
                # Add extracted variables to context
                for key, value in extracted_variables.items():
                    if key != 'suggested_question':  # Skip the suggested question key
                        # Skip optional variables if user says "No" or similar
                        if key in ["recent_medical_history", "allergies", "chronic_diseases"] and str(value).lower() in ["no", "nope", "none", "nothing", "n/a", "na", "n"]:
                            chat_context[user_id]["collected_variables"][key] = "None reported"
                        else:
                            chat_context[user_id]["collected_variables"][key] = value
                
                # Check what variables are still missing
                missing_variables = []
                collected = chat_context[user_id]["collected_variables"]
                for variable in REQUIRED_VARIABLES:
                    if variable not in collected or not collected[variable]:
                        missing_variables.append(variable)
                
                # If all required variables are collected, provide the medical response
                if not missing_variables:
                    chat_context[user_id]["variables_collected"] = True
                    return provide_medical_response(user_id, original_message, response_language, source_language)
                
                # Ask for missing variables one by one
                next_variable = get_next_uncollected_variable(user_id)
                if next_variable:
                    # Check if we already have this information
                    if next_variable in chat_context[user_id]["collected_variables"] and chat_context[user_id]["collected_variables"][next_variable]:
                        # Skip this variable and get the next one
                        next_variable = get_next_uncollected_variable(user_id)
                        if next_variable is None:
                            # All variables collected, provide medical response
                            chat_context[user_id]["variables_collected"] = True
                            return provide_medical_response(user_id, original_message, response_language, source_language)
                    
                    # Ask Gemini to generate a question for this specific variable
                    question = generate_variable_question_with_gemini(user_id, next_variable, user_message)
                    chat_context[user_id]["current_variable"] = next_variable
                    # Add AI response to chat history
                    add_to_chat_history(user_id, "assistant", question)
                    return Response(question, mimetype='text/plain')
                else:
                    # All variables collected, provide medical response
                    chat_context[user_id]["variables_collected"] = True
                    return provide_medical_response(user_id, original_message, response_language, source_language)
        else:
            # Initialize context and start collecting variables
            if user_id not in chat_context:
                initialize_user_context(user_id)
            
            # Extract all variables at once from the initial message
            extracted_variables = extract_variables_with_gemini(user_id, user_message)
            
            # Add extracted variables to context
            for key, value in extracted_variables.items():
                if key != 'suggested_question':  # Skip the suggested question key
                    # Skip optional variables if user says "No" or similar
                    if key in ["recent_medical_history", "allergies", "chronic_diseases"] and str(value).lower() in ["no", "nope", "none", "nothing", "n/a", "na", "n"]:
                        chat_context[user_id]["collected_variables"][key] = "None reported"
                    else:
                        chat_context[user_id]["collected_variables"][key] = value
            
            # Check what variables are still missing
            missing_variables = []
            collected = chat_context[user_id]["collected_variables"]
            for variable in REQUIRED_VARIABLES:
                if variable not in collected or not collected[variable]:
                    missing_variables.append(variable)
            
            # If all required variables are collected, provide the medical response
            if not missing_variables:
                chat_context[user_id]["variables_collected"] = True
                return provide_medical_response(user_id, original_message, response_language, source_language)
            
            # Ask for missing variables one by one
            next_variable = get_next_uncollected_variable(user_id)
            if next_variable:
                # Check if we already have this information
                if next_variable in chat_context[user_id]["collected_variables"] and chat_context[user_id]["collected_variables"][next_variable]:
                    # Skip this variable and get the next one
                    next_variable = get_next_uncollected_variable(user_id)
                    if next_variable is None:
                        # All variables collected, provide medical response
                        chat_context[user_id]["variables_collected"] = True
                        return provide_medical_response(user_id, original_message, response_language, source_language)
                
                # Ask Gemini to generate a question for this specific variable
                question = generate_variable_question_with_gemini(user_id, next_variable, user_message)
                chat_context[user_id]["current_variable"] = next_variable
                # Add AI response to chat history
                add_to_chat_history(user_id, "assistant", question)
                return Response(question, mimetype='text/plain')
            else:
                # All variables collected, provide medical response
                chat_context[user_id]["variables_collected"] = True
                return provide_medical_response(user_id, original_message, response_language, source_language)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        # Use fallback response for general errors
        try:
            data = request.get_json()
            user_message = data.get('message', '') if data else ''
            source_language = data.get('source_language', 'en') if data else 'en'
            user_id = data.get('user_id', 'default_user') if data else 'default_user'
            
            # Preprocess the message to English for medical query detection
            processed_message = user_message
            if source_language != 'en':
                processed_message = translate_text(user_message, 'en')
            
            is_medical = is_medical_query(processed_message) if processed_message else False
            error_response = "I'm currently experiencing technical difficulties. Please try again later."
            
            target_language = data.get('language', 'en') if data else 'en'
            translate_to = data.get('translate_to', 'en') if data else 'en'
            response_language = translate_to
            
            # No limits on translation in error handling either
            if response_language != 'en':
                error_response = translate_text(error_response, response_language)
            
            # Add AI response to chat history
            add_to_chat_history(user_id, "assistant", error_response)
            return Response(error_response, mimetype='text/plain')
        except Exception as inner_e:
            logger.error(f"Error in fallback error handling: {str(inner_e)}")
            error_msg = f"Service error: {str(e)}. Please try again later."
            target_language = request.get_json().get('language', 'en') if request.get_json() else 'en'
            translate_to = request.get_json().get('translate_to', 'en') if request.get_json() else 'en'
            response_language = translate_to
            user_id = request.get_json().get('user_id', 'default_user') if request.get_json() else 'default_user'
            # No limits on translation even for error messages
            if response_language != 'en':
                error_msg = translate_text(error_msg, response_language)
                logger.info(f"Translated error to {response_language}: {error_msg}")
            # Add AI response to chat history
            add_to_chat_history(user_id, "assistant", error_msg)
            return Response(error_msg, mimetype='text/plain')

# Endpoint for translating text to a target language
@app.route('/translate', methods=['POST'])
def translate_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('target_language', 'en')
        
        # Remove the text requirement limit - allow empty text to be handled gracefully
        # Just return empty text if that's what was sent
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
        
        # Remove the text requirement limit - allow empty text to be handled gracefully
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

# New endpoint for text-to-speech using Web Speech API directly
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        message_id = data.get('messageId', '')
        
        if not text:
            logger.warning("Text-to-speech request failed: No text provided")
            return jsonify({"error": "Text is required"}), 400
            
        # For Web Speech API, we return a JSON response with the text
        # The frontend will use the browser's Web Speech API to synthesize the audio
        # Modified to always use Heera voice for all languages
        response_data = {
            "text": text,
            "lang": "hi-IN",  # Always use Hindi language for Heera voice
            "voice": "Microsoft Heera"  # Always use Heera voice
        }
        
        # Return JSON response for Web Speech API handling in frontend
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Text-to-speech error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": "Text-to-speech service error", "details": str(e)}), 500

# New endpoint for stopping speech (placeholder for future implementation)
@app.route('/stop-speech', methods=['POST'])
def stop_speech():
    try:
        data = request.get_json()
        message_id = data.get('messageId', '')
        
        # In a real implementation, this would stop the audio playback
        # For now, we'll just return success
        return jsonify({
            "success": True,
            "messageId": message_id,
            "status": "stopped"
        })
        
    except Exception as e:
        logger.error(f"Stop speech error: {str(e)}")
        return jsonify({"error": f"Stop speech service error: {str(e)}"}), 500

def search_indian_medicines_for_symptoms(symptoms, max_results=5):
    """
    Search for relevant Indian medicines based on symptoms.
    
    Args:
        symptoms (str): Symptom description to search for
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of relevant medicines
    """
    if not symptoms or not INDIAN_MEDICINES:
        return []
    
    # Convert symptoms to lowercase for matching
    symptoms_lower = symptoms.lower()
    
    # Keywords that might indicate the type of medicine needed with common dosages
    # Prioritize fever and cold medicines for better matching
    symptom_keywords = {
        'fever': ['paracetamol', 'ibuprofen', 'crocin', 'calpol', 'fever', 'temperature'],
        'cold': ['cold', 'nasal', 'decongestant', 'cough', 'runny nose', 'congestion'],
        'cough': ['cough', 'expectorant', 'syrup', 'guaifenesin', 'dextromethorphan', 'chest'],
        'headache': ['headache', 'pain', 'analgesic', 'paracetamol'],
        'stomach': ['stomach', 'digestion', 'antacid', 'gastric', 'pantoprazole', 'ranitidine'],
        'allergy': ['allergy', 'antihistamine', 'itch', 'loratadine', 'cetirizine'],
        'infection': ['antibiotic', 'infection', 'amoxicillin', 'azithromycin', 'cefixime'],
        'pain': ['pain', 'analgesic', 'ache', 'diclofenac', 'ibuprofen'],
        'diarrhea': ['diarrhea', 'loperamide', 'stomach', 'racecadotril'],
        'constipation': ['constipation', 'laxative', 'fiber', 'lactulose'],
        'hypertension': ['hypertension', 'blood pressure', 'amlodipine', 'telmisartan'],
        'diabetes': ['diabetes', 'blood sugar', 'metformin', 'glimepiride']
    }
    
    # Find relevant medicines
    relevant_medicines = []
    
    # Score each medicine based on symptom matching
    for medicine in INDIAN_MEDICINES:
        score = 0
        medicine_text = f"{medicine['name']} {medicine['short_composition1']} {medicine['short_composition2']}".lower()
        
        # Check for symptom keywords in medicine information
        for symptom_key, keywords in symptom_keywords.items():
            if symptom_key in symptoms_lower:
                for keyword in keywords:
                    if keyword in medicine_text:
                        score += 3  # Higher score for direct matches
        
        # Also check if symptom words appear in medicine information
        symptom_words = symptoms_lower.split()
        for word in symptom_words:
            if len(word) > 3 and word in medicine_text:
                score += 1
                
        # Add medicine if it has a relevant score
        if score > 0:
            relevant_medicines.append((medicine, score))
    
    # Sort by score and return top results
    relevant_medicines.sort(key=lambda x: x[1], reverse=True)
    return [medicine for medicine, score in relevant_medicines[:max_results]]

def get_contextual_medicine_suggestions(symptoms):
    """
    Get medicine suggestions based on symptoms for use in the AI prompt.
    
    Args:
        symptoms (str): Symptom description
        
    Returns:
        str: Formatted medicine suggestions with dosage examples
    """
    if not symptoms:
        return ""
    
    # Search for relevant medicines
    relevant_medicines = search_indian_medicines_for_symptoms(symptoms, max_results=5)
    
    if not relevant_medicines:
        return ""
    
    # Format suggestions with dosage examples in a way that's easy for AI to incorporate
    suggestions = "EXAMPLE MEDICAL TREATMENT FORMAT FOR FEVER AND COLD:\n"
    suggestions += "Medical Treatment (5 points):\n"
    count = 1
    for medicine in relevant_medicines:
        # Determine dosage based on medicine type
        dosage = ""
        if 'syrup' in medicine['pack_size_label'].lower():
            dosage = "5ml to 10ml three times a day"
        elif 'tablet' in medicine['pack_size_label'].lower() or 'capsule' in medicine['pack_size_label'].lower():
            dosage = "1 tablet/capsule three times a day"
        elif 'injection' in medicine['pack_size_label'].lower():
            dosage = "as prescribed by doctor"
        elif 'cream' in medicine['pack_size_label'].lower() or 'gel' in medicine['pack_size_label'].lower():
            dosage = "apply to affected area three times a day"
        elif 'drop' in medicine['pack_size_label'].lower():
            dosage = "1-2 drops three times a day"
        elif 'inhaler' in medicine['pack_size_label'].lower():
            dosage = "1-2 puffs three times a day"
        else:
            dosage = "as directed by physician"
        
        # Get the primary use of the medicine based on its composition
        primary_use = ""
        composition = medicine['short_composition1'].lower()
        if 'paracetamol' in composition or 'acetaminophen' in composition:
            primary_use = "fever and pain relief"
        elif 'ibuprofen' in composition or 'diclofenac' in composition:
            primary_use = "anti-inflammatory and pain relief"
        elif 'azithromycin' in composition or 'amoxicillin' in composition:
            primary_use = "bacterial infection treatment"
        elif 'loratadine' in composition or 'cetirizine' in composition:
            primary_use = "allergy symptom relief"
        elif 'omeprazole' in composition or 'pantoprazole' in composition:
            primary_use = "acid reflux and stomach protection"
        elif 'levocetirizine' in composition:
            primary_use = "severe allergy and cold symptom relief"
        elif 'montelukast' in composition:
            primary_use = "asthma and allergy management"
        else:
            # Extract from medicine name if possible
            medicine_name = medicine['name'].lower()
            if 'cold' in medicine_name or 'flu' in medicine_name:
                primary_use = "cold and flu symptom relief"
            elif 'pain' in medicine_name or 'relief' in medicine_name:
                primary_use = "general pain relief"
            elif 'fever' in medicine_name:
                primary_use = "fever reduction"
            else:
                primary_use = "symptom relief"
        
        # Format the suggestion with explanation
        suggestions += f"- Take {medicine['name']} by {medicine['manufacturer_name']} ({medicine['short_composition1']}) - {dosage} for 5-7 days for {primary_use} - {medicine['short_composition1']} works by targeting the underlying cause of your symptoms\n"
        count += 1
        if count > 5:
            break
    
    return suggestions

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

# Vercel requires the app to be exported as `app`
# Keep the if __name__ == '__main__' block for local testing
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
