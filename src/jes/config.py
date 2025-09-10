#!/usr/bin/env python3
"""
SAS Viya Configuration

Configuration settings for SAS Viya OAuth authentication.
Loads settings from environment variables using dotenv.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SAS Viya Configuration - loaded from environment variables
SAS_CONFIG = {
    # Base URL of your SAS Viya environment
    "base_url": os.getenv("SAS_BASE_URL", "https://xarprodviya.ondemand.sas.com"),
    
    # OAuth Client ID
    "client_id": os.getenv("SAS_CLIENT_ID", "xar_api"),
    
    # User credentials (loaded from environment for security)
    "username": os.getenv("SAS_USERNAME"),
    "password": os.getenv("SAS_PASSWORD"),
    
    # OAuth settings
    "scope": os.getenv("SAS_SCOPE", "openid"),
    
    # Token storage
    "token_file": os.getenv("SAS_TOKEN_FILE", "data/sas_tokens.json"),
    
    # Tokens (if available in environment)
    "access_token": os.getenv("SAS_ACCESS_TOKEN"),
    "refresh_token": os.getenv("SAS_REFRESH_TOKEN"),
    "token_expires_at": os.getenv("SAS_TOKEN_EXPIRES_AT"),
    "refresh_expires_at": os.getenv("SAS_REFRESH_EXPIRES_AT"),
    
    # Test endpoint for token validation
    "test_endpoint": os.getenv("SAS_TEST_ENDPOINT", "/compute/contexts")
}

def get_config():
    """Get the SAS configuration with validation"""
    config = SAS_CONFIG.copy()
    
    # Validate required environment variables
    required_vars = ["username", "password"]
    missing_vars = [var for var in required_vars if not config.get(var)]
    
    if missing_vars:
        missing_env_vars = [f"SAS_{var.upper()}" for var in missing_vars]
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}. "
            f"Please create a .env file or set these environment variables."
        )
    
    return config

def update_config(**kwargs):
    """Update configuration values"""
    SAS_CONFIG.update(kwargs)

def validate_env_file():
    """Check if .env file exists and has required variables"""
    import os
    
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Please create one using env.template as a guide.")
        return False
    
    try:
        config = get_config()
        print("✅ Environment configuration loaded successfully")
        return True
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    import json
    print("Current SAS Viya Configuration:")
    print(json.dumps(SAS_CONFIG, indent=2))
