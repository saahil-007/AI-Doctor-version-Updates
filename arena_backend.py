import os
import logging
import re
import json
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from collections import deque
import requests
import google.generativeai as genai  # pyright: ignore[reportMissingImports]

# Load environment variables FIRST, before any other operations
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model availability
gemini_available = False
gemini_model = None
openrouter_available = False
openrouter_client = None

# Context window to store chat history (25 chats as per requirement)
arena_context = {
    'gemini': {},
    'gpt': {},
    'claude': {}
}

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

# Track if models have been initialized
gemini_initialized = False
openrouter_initialized = False

def initialize_gemini():
    global gemini_available, gemini_model, gemini_initialized
    
    # Only initialize if not already initialized
    if gemini_initialized:
        return
        
    try:
        # Configure the client with API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Validate API key format
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        if len(api_key) < 30:
            raise ValueError("GOOGLE_API_KEY appears to be invalid (too short)")
            
        # Initialize the client
        genai.configure(api_key=api_key)  # pyright: ignore[reportPrivateImportUsage]
        
        # Create a single instance of the Gemini model for reuse
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')  # pyright: ignore[reportPrivateImportUsage]
        
        print("Google GenAI client configured successfully")
        logger.info("Google GenAI client configured successfully")
        
        # Test the client with a simple request
        print("Testing client accessibility...")
        test_response = gemini_model.generate_content("Explain how AI works in a few words")
        print(f"Test response: {test_response.text}")
        print("Client initialized successfully")
        logger.info("Client initialized successfully")
        gemini_available = True
        gemini_initialized = True

    except Exception as e:
        print(f"ERROR: Google GenAI client configuration error: {str(e)}")
        logger.error(f"Google GenAI client configuration error: {str(e)}")
        gemini_available = False

def initialize_openrouter():
    global openrouter_available, openrouter_client, openrouter_initialized
    
    # Only initialize if not already initialized
    if openrouter_initialized:
        return
        
    try:
        # Configure the client with API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Validate API key format
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        if len(api_key) < 30:
            raise ValueError("OPENROUTER_API_KEY appears to be invalid (too short)")
            
        # Initialize the client
        openrouter_client = requests.Session()
        openrouter_client.headers.update({
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:5000",  # For openrouter
            "X-Title": "AI Doctor",  # For openrouter
            "Content-Type": "application/json"
        })
        
        print("OpenRouter client configured successfully")
        logger.info("OpenRouter client configured successfully")
        
        # Test the client with a simple request
        print("Testing client accessibility...")
        test_response = openrouter_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Explain how AI works in a few words"}],
                "max_tokens": 50
            }
        )
        
        if test_response.status_code == 200:
            print("Client initialized successfully")
            logger.info("Client initialized successfully")
            openrouter_available = True
            openrouter_initialized = True
        else:
            raise Exception(f"OpenRouter test failed with status {test_response.status_code}")

    except Exception as e:
        print(f"ERROR: OpenRouter client configuration error: {str(e)}")
        logger.error(f"OpenRouter client configuration error: {str(e)}")
        openrouter_available = False

# Don't initialize models at startup - only initialize when endpoints are called
print("Models will be initialized on demand when /arena endpoints are accessed")

# Language codes mapping
LANGUAGE_CODES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi'
}

# Import the translate_text function from the new module
from translate import translate_text

# Load system prompts for each model
def load_system_prompts():
    prompts = {}
    try:
        # Load Gemini prompt
        with open('prompts/gemini_prompt.txt', 'r', encoding='utf-8') as f:
            prompts['gemini'] = f.read()
            
        # Load GPT prompt
        with open('prompts/gpt_prompt.txt', 'r', encoding='utf-8') as f:
            prompts['gpt'] = f.read()
            
        # Load Claude prompt
        with open('prompts/claude_prompt.txt', 'r', encoding='utf-8') as f:
            prompts['claude'] = f.read()
            
        return prompts
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {str(e)}")
        # Fallback prompts
        fallback_prompt = """You are Dr. Vaani, a friendly AI medical assistant. Provide helpful, direct answers to medical questions.
        
User concern: {user_message}
        
Dr. Vaani:"""
        return {
            'gemini': fallback_prompt,
            'gpt': fallback_prompt,
            'claude': fallback_prompt
        }

# Load system prompts
SYSTEM_PROMPTS = load_system_prompts()

def extract_variables_with_model(user_id, message, model_type='gemini'):
    """Use the specified model to extract all variables from user message and chat history with high accuracy"""
    global gemini_model, openrouter_client
    
    # Use the specified model to extract variables
    try:
        # Build context from chat history
        context_parts = []
        context_to_use = arena_context.get(model_type, {})
        if user_id in context_to_use and context_to_use[user_id]["chat_history"]:
            # Get last few messages for context (maintaining 25-message window)
            recent_messages = list(context_to_use[user_id]["chat_history"])[-25:]
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Dr. Vaani"
                context_parts.append(f"{role}: {msg['message']}")
        
        context = "\n".join(context_parts)
        
        # Create prompt for variable extraction
        # For arena mode, only extract age, gender, and symptoms_duration
        if model_type in ['gemini', 'gpt', 'claude']:
            prompt = f"""
You are an expert medical information extractor and conversational assistant. Your task is to analyze the user's message and extract ONLY the following three medical variables with extreme precision from a SINGLE message:

1. age: Patient's age in years (number only) - Look for phrases like "my X year old son/daughter", "X years old", "I'm 20 years old", "20 year old", "20 yrs", etc.
2. gender: Patient's gender (Male/Female/Other) - Look for words like "male", "female", "man", "woman", "boy", "girl", "son" (implies Male), or "daughter" (implies Female)
3. symptoms_duration: How long the patient has been experiencing symptoms (be very specific, e.g., "since yesterday", "3 days", "since this morning")

Context (conversation history):
{context}

Current user message: {message}

CRITICAL INSTRUCTIONS:
- Extract ONLY the three variables listed above (age, gender, symptoms_duration)
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

Example response formats:
For message "im 20 , male":
{{
  "age": "20",
  "gender": "Male"
}}

For message "my 5 year old son has fever since yesterday":
{{
  "age": "5",
  "gender": "Male",
  "symptoms_duration": "since yesterday"
}}

For message "hello my 7 year daughter is having running nose for 2 days":
{{
  "age": "7",
  "gender": "Female",
  "symptoms_duration": "for 2 days"
}}

For message "I have headache since this morning":
{{
  "symptoms_duration": "since this morning"
}}

JSON Response:"""
        else:
            # For chat mode, extract all variables
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
        
        # Use the appropriate model
        if model_type == 'gemini' and gemini_model is not None:
            response = gemini_model.generate_content(prompt)
            response_text = response.text
        elif model_type in ['gpt', 'claude'] and openrouter_client is not None:
            model_map = {
                'gpt': 'openai/gpt-3.5-turbo',
                'claude': 'anthropic/claude-3.5-sonnet:free'
            }
            
            response = openrouter_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": model_map.get(model_type, 'openai/gpt-3.5-turbo'),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.1
                }
            )
            
            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenRouter request failed with status {response.status_code}")
        else:
            raise Exception(f"{model_type} model not available")
        
        # Parse the JSON response
        import json
        try:
            extracted = json.loads(response_text.strip())
            return extracted
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty dict
            logger.warning(f"{model_type} variable extraction failed")
            return {}
            
    except Exception as e:
        logger.error(f"Error in {model_type} variable extraction: {str(e)}")
        # Return empty dict if model fails
        return {}

def generate_variable_question_with_model(user_id, variable_name, user_message, model_type='gemini'):
    """Use the specified model to generate a contextual question for a specific variable"""
    global gemini_model, openrouter_client
    
    # Use the specified model to generate the question
    try:
        # Build context from chat history
        context_parts = []
        context_to_use = arena_context.get(model_type, {})
        if user_id in context_to_use and context_to_use[user_id]["chat_history"]:
            # Get last few messages for context (maintaining 25-message window)
            recent_messages = list(context_to_use[user_id]["chat_history"])[-25:]
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Dr. Vaani"
                context_parts.append(f"{role}: {msg['message']}")
        
        context = "\n".join(context_parts)
        
        # Create prompt for question generation
        # For arena mode, be more direct and specific
        if model_type in ['gemini', 'gpt', 'claude']:
            prompt = f"""
You are an expert medical conversational assistant. Your task is to generate a direct, clear question to collect a specific piece of medical information from the user.

The variable you need to ask about is: {variable_name}

Context (conversation history):
{context}

Current user message: {user_message}

Generate a direct, clear question to collect information about {variable_name}. 
The question should be:
1. Direct and clear, not conversational
2. Specific to the variable being collected
3. Easy for the user to understand and answer
4. Should NOT ask for information that is already provided in the context

Example outputs:
- For age: "What is your age?"
- For gender: "What is your gender?"
- For symptoms_duration: "How long have you been experiencing these symptoms?"

Question:"""
        else:
            # For chat mode, use the original conversational approach
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
        
        # Use the appropriate model
        if model_type == 'gemini' and gemini_model is not None:
            response = gemini_model.generate_content(prompt)
            question = response.text.strip()
        elif model_type in ['gpt', 'claude'] and openrouter_client is not None:
            model_map = {
                'gpt': 'openai/gpt-3.5-turbo',
                'claude': 'anthropic/claude-3.5-sonnet:free'
            }
            
            response = openrouter_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": model_map.get(model_type, 'openai/gpt-3.5-turbo'),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                response_data = response.json()
                question = response_data['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"OpenRouter request failed with status {response.status_code}")
        else:
            raise Exception(f"{model_type} model not available")
        
        # Return the generated question
        return question if question else f"Please provide information about {variable_name.replace('_', ' ')}."
            
    except Exception as e:
        logger.error(f"Error in {model_type} question generation: {str(e)}")
        # Return a generic question if model fails
        return f"Could you please provide information about your {variable_name.replace('_', ' ')}?"

# Context window functions for variable collection
def initialize_user_context(user_id, context_type='chat'):
    """Initialize context for a new user"""
    context_to_use = arena_context.get(context_type, {})
    context_to_use[user_id] = {
        "chat_history": deque(maxlen=25),  # Keep last 25 messages as per requirement
        "collected_variables": {},
        "current_variable": None,
        "variables_collected": False
    }

def add_to_chat_history(user_id, role, message, context_type='chat'):
    """Add a message to the user's chat history"""
    context_to_use = arena_context.get(context_type, {})
    if user_id not in context_to_use:
        initialize_user_context(user_id, context_type)
    
    context_to_use[user_id]["chat_history"].append({
        "role": role,
        "message": message
    })

def get_next_uncollected_variable(user_id, context_type='chat'):
    """Get the next variable that needs to be collected"""
    context_to_use = arena_context.get(context_type, {})
    if user_id not in context_to_use:
        initialize_user_context(user_id, context_type)
    
    collected = context_to_use[user_id]["collected_variables"]
    
    # Use simplified required variables for arena mode
    if context_type in ['gemini', 'gpt', 'claude']:
        required_vars = ["age", "gender", "symptoms_duration"]
    else:
        required_vars = ["age", "gender", "symptoms_duration"]
    
    for variable in required_vars:
        if variable not in collected or not collected[variable]:
            return variable
    
    # Then check other variables (only for chat mode)
    if context_type == 'chat':
        for variable in REQUIRED_VARIABLES:
            if variable not in collected or not collected[variable]:
                return variable
    
    return None

def collect_variable(user_id, variable, value, context_type='chat'):
    """Collect a variable value from the user"""
    context_to_use = arena_context.get(context_type, {})
    if user_id not in context_to_use:
        initialize_user_context(user_id, context_type)
    
    # Skip optional variables if user says "No" or similar
    if variable in ["recent_medical_history", "allergies", "chronic_diseases"] and value.lower() in ["no", "nope", "none", "nothing", "n/a", "na", "n"]:
        value = "None reported"
    
    context_to_use[user_id]["collected_variables"][variable] = value
    
    # Check if all required variables are collected
    next_var = get_next_uncollected_variable(user_id, context_type)
    if next_var is None:
        context_to_use[user_id]["variables_collected"] = True
    
    return next_var

def get_user_context_prompt(user_id, context_type='chat'):
    """Generate a prompt with the collected user context"""
    context_to_use = arena_context.get(context_type, {})
    if user_id not in context_to_use or not context_to_use[user_id]["variables_collected"]:
        return ""
    
    variables = context_to_use[user_id]["collected_variables"]
    context_parts = []
    
    for variable in REQUIRED_VARIABLES:
        if variable in variables and variables[variable]:
            # Format variable name for readability
            formatted_name = variable.replace("_", " ").title()
            context_parts.append(f"{formatted_name}: {variables[variable]}")
    
    if context_parts:
        return "\n\nPatient Information:\n" + "\n".join(context_parts)
    
    return ""

def reset_user_context(user_id, context_type='chat'):
    """Reset the user context after providing a medical response"""
    context_to_use = arena_context.get(context_type, {})
    if user_id in context_to_use:
        context_to_use[user_id]["collected_variables"] = {}
        context_to_use[user_id]["current_variable"] = None
        context_to_use[user_id]["variables_collected"] = False

def provide_medical_response(user_id, original_message, response_language, source_language, model_type='gemini'):
    """Provide the final structured medical response after collecting all variables"""
    global gemini_model, openrouter_client
    
    # Use the system prompt for the specific model
    system_prompt = SYSTEM_PROMPTS.get(model_type, SYSTEM_PROMPTS['gemini'])
    
    # Prepare the prompt with system context
    # Use the ORIGINAL message for the AI to maintain context in the user's language
    prompt = system_prompt.format(user_message=original_message)
    
    # Add patient information context
    patient_context = get_user_context_prompt(user_id, model_type)
    if patient_context:
        prompt += patient_context
    
    # Extract symptoms from the context to suggest relevant medicines
    symptoms_info = ""
    context_to_use = arena_context.get(model_type, {})
    if user_id in context_to_use and "collected_variables" in context_to_use[user_id]:
        variables = context_to_use[user_id]["collected_variables"]
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
    
    logger.info(f"Sending prompt to {model_type}: {prompt[:100]}...")  # Log first 100 chars
    
    # Generate response using the specified model
    try:
        if model_type == 'gemini' and gemini_model is not None:
            response = gemini_model.generate_content(prompt)
            full_response = response.text if hasattr(response, 'text') else str(response)
        elif model_type in ['gpt', 'claude'] and openrouter_client is not None:
            model_map = {
                'gpt': 'openai/gpt-3.5-turbo',
                'claude': 'anthropic/claude-3.5-sonnet:free'
            }
            
            response = openrouter_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": model_map.get(model_type, 'openai/gpt-3.5-turbo'),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                response_data = response.json()
                full_response = response_data['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenRouter request failed with status {response.status_code}")
        else:
            raise Exception(f"{model_type} model not available")
    except Exception as e:
        logger.error(f"Error generating content with {model_type}: {str(e)}")
        # Return error response when model fails
        error_response = "I'm currently experiencing technical difficulties. Please try again later."
        # Add AI response to chat history
        add_to_chat_history(user_id, "assistant", error_response, model_type)
        # Reset user context
        reset_user_context(user_id, model_type)
        return Response(error_response, mimetype='text/plain')
    
    # Clean the response
    cleaned_response = clean_response(full_response)
    
    # Add AI response to chat history
    add_to_chat_history(user_id, "assistant", cleaned_response, model_type)
    
    # Log the complete response
    logger.info(f"Complete response from {model_type}: {cleaned_response[:100]}...")
    
    # Reset user context for next consultation
    reset_user_context(user_id, model_type)
    
    # Translate the entire response if needed - no limits on translation
    if response_language != 'en':
        translated_response = translate_text(cleaned_response, response_language)
        logger.info(f"Translated response to {response_language}: {translated_response[:100]}...")
        return Response(translated_response, mimetype='text/plain')
    
    return Response(cleaned_response, mimetype='text/plain')

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

def create_arena_app():
    """Create Flask app for arena mode"""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Increase to 50MB limit for larger requests
    
    # Updated CORS configuration to allow requests from Vercel deployment
    # Allow all origins to simplify deployment across Vercel preview and production URLs
    CORS(app, origins="*", supports_credentials=False)
    
    @app.route('/arena', methods=['POST'])
    def arena():
        # Initialize all models only when /arena is accessed
        initialize_gemini()
        initialize_openrouter()
        
        # Initialize variables at the start to avoid possibly unbound errors
        response_language = 'en'
        
        try:
            # Get user message and language from request
            data = request.get_json()
            user_message = data.get('message', '') if data else ''
            target_language = data.get('language', 'en') if data else 'en'
            translate_to = data.get('translate_to', 'en') if data else 'en'
            source_language = data.get('source_language', 'en') if data else 'en'
            user_id = data.get('user_id', 'arena_user') if data else 'arena_user'
            
            # Log the translation parameters for debugging
            logger.info(f"Arena - Received translation parameters - target_language: {target_language}, translate_to: {translate_to}, source_language: {source_language}")
            
            # For dynamic translation, we want to translate the AI response to the selected language
            # translate_to is the language we want to translate the response to
            response_language = translate_to
            
            if not user_message:
                return jsonify({"error": "Message is required"}), 400
            
            # Store the original message for AI processing
            original_message = user_message
            
            # Add user message to all arena chat histories
            add_to_chat_history(user_id, "user", user_message, 'gemini')
            add_to_chat_history(user_id, "user", user_message, 'gpt')
            add_to_chat_history(user_id, "user", user_message, 'claude')
            
            # Preprocess the message to English for better medical query detection
            processed_message = user_message
            if source_language != 'en':
                try:
                    # Translate to English for medical keyword detection
                    processed_message = translate_text(user_message, 'en')
                    logger.info(f"Arena - Preprocessed message from {source_language} to English: '{user_message}' -> '{processed_message}'")
                except Exception as preprocess_error:
                    logger.error(f"Arena - Preprocessing error: {str(preprocess_error)}")
                    # Continue with original message if preprocessing fails
                    processed_message = user_message
            
            # Check if this is a simple greeting
            if is_simple_greeting(processed_message):
                # Give each model its own unique greeting response
                greetings = {
                    'gemini': "Hello! I'm Dr. Vaani from Gemini. I'm here to provide you with medical assistance. What health concerns would you like to discuss today?",
                    'gpt': "Hi there! I'm your GPT medical assistant. I'm ready to help with any health questions you might have. What can I assist you with?",
                    'claude': "Greetings! I'm Claude, your AI health companion. I'm here to offer medical guidance and support. What would you like to know?"
                }
                
                # Add AI responses to chat histories
                for model_type in ['gemini', 'gpt', 'claude']:
                    add_to_chat_history(user_id, "assistant", greetings[model_type], model_type)
                
                # Translate responses if needed
                if response_language != 'en':
                    for model_type in greetings:
                        greetings[model_type] = translate_text(greetings[model_type], response_language)
                        logger.info(f"Arena - Translated {model_type} greeting to {response_language}: {greetings[model_type]}")
                
                return jsonify(greetings)
            
            # Check if this is a medical query using the preprocessed (English) message
            is_medical = is_medical_query(processed_message)
            
            # If we're already in the middle of collecting medical variables in any model,
            # continue with medical consultation regardless of current message content
            gemini_collecting = user_id in arena_context['gemini'] and not arena_context['gemini'][user_id].get("variables_collected", False)
            gpt_collecting = user_id in arena_context['gpt'] and not arena_context['gpt'][user_id].get("variables_collected", False)
            claude_collecting = user_id in arena_context['claude'] and not arena_context['claude'][user_id].get("variables_collected", False)
            
            if gemini_collecting or gpt_collecting or claude_collecting:
                is_medical = True
            
            # If not a medical query, provide general response from all models
            if not is_medical:
                # Prepare responses from all models
                responses = {}
                
                # Get response from Gemini
                if gemini_available and gemini_model is not None:
                    system_prompt = SYSTEM_PROMPTS.get('gemini', '')
                    prompt = system_prompt.format(user_message=original_message)
                    prompt += "\n\nIMPORTANT: User wants general conversation. Respond naturally and briefly. Be helpful but don't provide medical advice unless explicitly asked."
                    prompt += f"\n\nIMPORTANT: Respond in the same language as the user's message: {original_message}"
                    
                    response = gemini_model.generate_content(prompt)
                    responses['gemini'] = response.text if hasattr(response, 'text') else str(response)
                else:
                    responses['gemini'] = "Sorry, Gemini is not available right now."
                
                # Get response from GPT
                if openrouter_client is not None:
                    system_prompt = SYSTEM_PROMPTS.get('gpt', '')
                    prompt = system_prompt.format(user_message=original_message)
                    prompt += "\n\nIMPORTANT: User wants general conversation. Respond naturally and briefly. Be helpful but don't provide medical advice unless explicitly asked."
                    prompt += f"\n\nIMPORTANT: Respond in the same language as the user's message: {original_message}"
                    
                    response = openrouter_client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json={
                            "model": "openai/gpt-3.5-turbo",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 500,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        responses['gpt'] = response_data['choices'][0]['message']['content']
                    else:
                        responses['gpt'] = "Sorry, GPT is not available right now."
                else:
                    responses['gpt'] = "Sorry, OpenRouter is not available right now."
                
                # Get response from Claude
                if openrouter_client is not None:
                    system_prompt = SYSTEM_PROMPTS.get('claude', '')
                    prompt = system_prompt.format(user_message=original_message)
                    prompt += "\n\nIMPORTANT: User wants general conversation. Respond naturally and briefly. Be helpful but don't provide medical advice unless explicitly asked."
                    prompt += f"\n\nIMPORTANT: Respond in the same language as the user's message: {original_message}"
                    
                    response = openrouter_client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json={
                            "model": "anthropic/claude-3.5-sonnet:free",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 500,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        responses['claude'] = response_data['choices'][0]['message']['content']
                    else:
                        responses['claude'] = "Sorry, Claude is not available right now."
                else:
                    responses['claude'] = "Sorry, OpenRouter is not available right now."
                
                # Add AI responses to chat histories
                for model_type in ['gemini', 'gpt', 'claude']:
                    if model_type in responses:
                        add_to_chat_history(user_id, "assistant", responses[model_type], model_type)
                
                # Translate responses if needed
                if response_language != 'en':
                    for model_type in responses:
                        responses[model_type] = translate_text(responses[model_type], response_language)
                        logger.info(f"Arena - Translated {model_type} response to {response_language}: {responses[model_type][:100]}...")
                
                # Clean responses
                for model_type in responses:
                    responses[model_type] = clean_response(responses[model_type])
                
                # Ensure we return JSON
                return jsonify(responses)
            
            # Handle medical query with simplified variable collection for all models
            # First, extract variables from all models simultaneously
            extracted_variables = {}
            for model_type in ['gemini', 'gpt', 'claude']:
                extracted_variables[model_type] = extract_variables_with_model(user_id, user_message, model_type)
                logger.info(f"Arena - Extracted variables from {model_type}: {extracted_variables[model_type]}")
            
            # Add extracted variables to contexts for all models
            for model_type in ['gemini', 'gpt', 'claude']:
                if user_id not in arena_context[model_type]:
                    initialize_user_context(user_id, model_type)
                    
                for key, value in extracted_variables[model_type].items():
                    if key != 'suggested_question':  # Skip the suggested question key
                        # Only collect age, gender, and symptoms_duration for arena mode
                        if key in ["age", "gender", "symptoms_duration"]:
                            arena_context[model_type][user_id]["collected_variables"][key] = value
        
            # Check what variables are still missing for each model (only age, gender, symptoms_duration)
            missing_variables = {}
            for model_type in ['gemini', 'gpt', 'claude']:
                missing_variables[model_type] = []
                collected = arena_context[model_type][user_id]["collected_variables"]
                for variable in ["age", "gender", "symptoms_duration"]:
                    if variable not in collected or not collected[variable]:
                        missing_variables[model_type].append(variable)
                logger.info(f"Arena - Missing variables for {model_type}: {missing_variables[model_type]}")
        
            # If all required variables are collected for all models, provide medical responses
            all_collected = all(len(missing_variables[model_type]) == 0 for model_type in ['gemini', 'gpt', 'claude'])
            if all_collected:
                logger.info("Arena - All variables collected for all models, providing medical responses")
                responses = {}
                for model_type in ['gemini', 'gpt', 'claude']:
                    arena_context[model_type][user_id]["variables_collected"] = True
                    response = provide_medical_response(user_id, original_message, response_language, source_language, model_type)
                    # Extract text from Response object
                    responses[model_type] = response.get_data(as_text=True)
                # Ensure we return JSON
                return jsonify(responses)
            
            # Otherwise, ask for missing variables from each model independently
            questions = {}
            for model_type in ['gemini', 'gpt', 'claude']:
                next_variable = get_next_uncollected_variable(user_id, model_type)
                if next_variable:
                    # Check if we already have this information in the model's context
                    if next_variable in arena_context[model_type][user_id]["collected_variables"] and arena_context[model_type][user_id]["collected_variables"][next_variable]:
                        # Skip this variable and get the next one
                        next_variable = get_next_uncollected_variable(user_id, model_type)
                    
                    if next_variable:
                        # Ask model to generate a question for this specific variable
                        question = generate_variable_question_with_model(user_id, next_variable, user_message, model_type)
                        arena_context[model_type][user_id]["current_variable"] = next_variable
                        # Add AI response to chat history
                        add_to_chat_history(user_id, "assistant", question, model_type)
                        questions[model_type] = question
                    else:
                        # All variables collected for this model, provide medical response
                        arena_context[model_type][user_id]["variables_collected"] = True
                        response = provide_medical_response(user_id, original_message, response_language, source_language, model_type)
                        questions[model_type] = response.get_data(as_text=True)
                else:
                    # All variables collected for this model, provide medical response
                    arena_context[model_type][user_id]["variables_collected"] = True
                    response = provide_medical_response(user_id, original_message, response_language, source_language, model_type)
                    questions[model_type] = response.get_data(as_text=True)
        
            # If we have questions, return them as plain text (first model's question)
            # In a real implementation, you might want to handle this differently
            if questions:
                # Return the first available question
                for model_type in ['gemini', 'gpt', 'claude']:
                    if model_type in questions:
                        return Response(questions[model_type], mimetype='text/plain')
        
            # If no questions, all models should have provided medical responses
            return jsonify(questions)
        
        except Exception as e:
            logger.error(f"Error in arena endpoint: {str(e)}")
            # Return error responses for all models as JSON
            error_response = "I'm currently experiencing technical difficulties. Please try again later."
            # Always default to English if there are issues with translation
            if response_language != 'en':
                error_response = translate_text(error_response, response_language)
            return jsonify({
                'gemini': error_response,
                'gpt': error_response,
                'claude': error_response
            }), 500

    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint to verify environment variables"""
        import os
        gemini_key = os.getenv("GOOGLE_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        return {
            "status": "ok",
            "models": {
                "gemini": {
                    "available": gemini_available,
                    "api_key_exists": gemini_key is not None,
                    "api_key_length": len(gemini_key) if gemini_key else 0
                },
                "openrouter": {
                    "available": openrouter_available,
                    "api_key_exists": openrouter_key is not None,
                    "api_key_length": len(openrouter_key) if openrouter_key else 0
                }
            }
        }

    return app