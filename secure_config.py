"""
Secure Configuration Management
==============================
Handles environment variables and secure config loading for production deployment.
"""
import os
import secrets
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Secure configuration management class"""
    
    # Flask Security
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    
    # Admin Authentication
    ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')  # Must be set in .env
    
    # Alpaca API Configuration
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY') 
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # API Security
    WEBHOOK_API_KEY = os.getenv('WEBHOOK_API_KEY', secrets.token_urlsafe(32))
    
    # Production Settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    ENABLE_HTTPS_REDIRECT = os.getenv('ENABLE_HTTPS_REDIRECT', 'true').lower() == 'true'
    ENABLE_SECURITY_HEADERS = os.getenv('ENABLE_SECURITY_HEADERS', 'true').lower() == 'true'
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 10))
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', 100))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_TRADE_LOGGING = os.getenv('ENABLE_TRADE_LOGGING', 'true').lower() == 'true'
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        errors = []
        
        if not cls.ADMIN_PASSWORD:
            errors.append("ADMIN_PASSWORD must be set in environment")
            
        if not cls.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY must be set in environment")
            
        if not cls.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY must be set in environment")
            
        if cls.FLASK_ENV == 'production' and cls.SECRET_KEY == 'dev':
            errors.append("SECRET_KEY must be set for production")
            
        return errors
    
    @classmethod
    def get_alpaca_config(cls):
        """Get Alpaca API configuration in legacy format"""
        return {
            "sessions": {
                "main": {
                    "name": "Secure_Trading",
                    "api_key": cls.ALPACA_API_KEY,
                    "api_secret": cls.ALPACA_SECRET_KEY,
                    "base_url": cls.ALPACA_BASE_URL,
                    "enabled": True
                }
            },
            "default_session": "main"
        }

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path('.env')
    template_file = Path('.env.template')
    
    if not env_file.exists() and template_file.exists():
        # Copy template and generate secure defaults
        with open(template_file, 'r') as template:
            content = template.read()
            
        # Replace placeholders with secure defaults
        content = content.replace('your-super-secret-flask-key-change-this', secrets.token_hex(32))
        content = content.replace('your-webhook-api-key-for-external-access', secrets.token_urlsafe(32))
        
        with open(env_file, 'w') as env:
            env.write(content)
            
        print("Created .env file from template. Please update with your actual credentials.")
        return False
    
    return env_file.exists()
