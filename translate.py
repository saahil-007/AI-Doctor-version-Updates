import urllib.request
import urllib.parse
import json
import logging
import re

# Set up logging
logger = logging.getLogger(__name__)

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
            logger.info(f"Translated '{text[:5000]}...' to {target_language}: '{translated_text[:100]}...'")
            return translated_text
        else:
            logger.error(f"Translation API error: {data}")
            # Remove the text length limit - translate all texts regardless of length
            # Split text into smaller chunks and translate each
            # For word-to-word translation, split by sentences
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