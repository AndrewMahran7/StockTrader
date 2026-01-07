"""
Configuration Management for AWS Deployment
==========================================
Handles environment variables and secure config loading for cloud and local deployment.
"""
import os
import secrets
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for AWS and local deployment"""
    
    # Flask Security
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    
    # Alpaca API Configuration
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY') 
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    
    # Trading Configuration
    TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'TSLA')
    POLLING_SECONDS = int(os.getenv('POLLING_SECONDS', 60))
    USE_STREAMING = os.getenv('USE_STREAMING', 'true').lower() == 'true'
    
    # Environment Detection
    @classmethod
    def is_production(cls):
        """Check if running in production environment"""
        return any([
            os.environ.get('AWS_EXECUTION_ENV'),
            os.environ.get('AWS_LAMBDA_FUNCTION_NAME'),
            os.environ.get('ECS_CONTAINER_METADATA_URI'),
            os.path.exists('/.dockerenv'),
            os.environ.get('FLASK_ENV') == 'production'
        ])
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        errors = []
            
        if not cls.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY must be set in environment")
            
        if not cls.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY must be set in environment")
            
        return errors
    
    @classmethod
    def get_alpaca_config(cls):
        """Get Alpaca API configuration for sessions"""
        environment_name = "AWS_Trading" if cls.is_production() else "Local_Trading"
        
        return {
            "sessions": {
                "main": {
                    "name": environment_name,
                    "api_key": cls.ALPACA_API_KEY,
                    "api_secret": cls.ALPACA_SECRET_KEY,
                    "base_url": cls.ALPACA_BASE_URL,
                    "enabled": True
                }
            },
            "default_session": "main",
            "strategy": {
                "enabled": True,
                "symbol": cls.TRADING_SYMBOL,
                "polling_seconds": cls.POLLING_SECONDS,
                "use_streaming": cls.USE_STREAMING
            }
        }

def create_env_file():
    """Create .env file template for AWS deployment"""
    env_file = Path('.env')
    
    if not env_file.exists():
        template = """# Alpaca Trading API Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Flask Configuration  
SECRET_KEY=your_secret_key_here
LOG_LEVEL=INFO

# Trading Configuration
TRADING_SYMBOL=TSLA
POLLING_SECONDS=60
USE_STREAMING=false

# AWS Configuration (optional)
AWS_REGION=us-east-1
FLASK_ENV=production
"""
        env_file.write_text(template)
        print(f"✅ Created .env template at {env_file}")
        print("📝 Please update the .env file with your actual Alpaca API credentials")
        return False
    
    return env_file.exists()
