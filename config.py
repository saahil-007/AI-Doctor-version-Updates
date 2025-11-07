import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class."""
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
    
    # Flask settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # API settings
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 30))
    
    # Model configurations
    MODEL_CONFIGS = {
        "gemini": {
            "name": "gemini-2.5-flash",
            "api_key": GOOGLE_API_KEY,
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        },
        "gpt": {
            "name": "openai/gpt-4o-mini",
            "api_key": OPENROUTER_API_KEY,
            "endpoint": "https://openrouter.ai/api/v1/chat/completions"
        },
        "claude": {
            "name": "anthropic/claude-3.5-sonnet",
            "api_key": OPENROUTER_API_KEY,
            "endpoint": "https://openrouter.ai/api/v1/chat/completions"
        }
    }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}