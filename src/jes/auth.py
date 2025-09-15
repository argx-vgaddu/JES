#!/usr/bin/env python3
"""
SAS Viya Authentication Module

Consolidated authentication handling with automatic token refresh.
Combines functionality from sas_oauth.py and simple_token_refresh.py.
"""

import requests
import urllib.parse
import webbrowser
import time
import json
from datetime import datetime, timedelta
import os


class SASViyaAuth:
    """Consolidated SAS Viya authentication with automatic token management"""
    
    def __init__(self, base_url, client_id, username, password):
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.username = username
        self.password = password
        self.tokens = {}
        
        # OAuth endpoints
        self.auth_url = f"{self.base_url}/SASLogon/oauth/authorize"
        self.token_url = f"{self.base_url}/SASLogon/oauth/token"
        
    def get_authorization_code(self):
        """Step 1: Get authorization code via browser (manual URL capture)"""
        print("🔐 Starting OAuth authorization flow...")
        
        # Build authorization URL
        auth_params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'scope': 'openid'
        }
        
        auth_url = f"{self.auth_url}?{urllib.parse.urlencode(auth_params)}"
        
        print(f"📋 Authorization URL: {auth_url}")
        print("🌐 Opening browser for authorization...")
        webbrowser.open(auth_url)
        
        print("\n" + "="*60)
        print("📌 MANUAL STEP REQUIRED:")
        print("1. Complete the login process in your browser")
        print("2. After login, you'll be redirected to an error page")
        print("3. Look at the URL in your browser - it will contain 'code=...'")
        print("4. Copy the authorization code from the URL")
        print("="*60 + "\n")
        
        # Prompt user to enter the authorization code manually
        while True:
            try:
                auth_code = input("🔑 Please paste the authorization code from the URL: ").strip()
                
                if not auth_code:
                    print("❌ Please enter a valid authorization code")
                    continue
                
                # Basic validation
                if len(auth_code) < 10:
                    print("❌ Authorization code seems too short. Please check and try again.")
                    continue
                
                print(f"✅ Authorization code received: {auth_code[:20]}...")
                return auth_code
                
            except KeyboardInterrupt:
                print("\n❌ Authorization cancelled by user")
                raise
            except Exception as e:
                print(f"❌ Error reading authorization code: {e}")
                continue
    
    def exchange_code_for_tokens(self, authorization_code):
        """Step 2: Exchange authorization code for tokens"""
        print("🔄 Exchanging authorization code for tokens...")
        
        token_data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'client_id': self.client_id
        }
        
        auth = (self.username, self.password)
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.post(
                self.token_url,
                data=token_data,
                auth=auth,
                headers=headers,
                timeout=30
            )
            
            print(f"📡 Token request status: {response.status_code}")
            
            if response.status_code == 200:
                self.tokens = response.json()
                
                # Add expiration times
                now = datetime.now()
                if 'expires_in' in self.tokens:
                    self.tokens['expires_at'] = (now + timedelta(seconds=self.tokens['expires_in'])).isoformat()
                if 'refresh_expires_in' in self.tokens:
                    self.tokens['refresh_expires_at'] = (now + timedelta(seconds=self.tokens['refresh_expires_in'])).isoformat()
                
                print("✅ Tokens received successfully!")
                return self.tokens
            else:
                print(f"❌ Token request failed: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during token exchange: {e}")
            raise
    
    def refresh_access_token(self):
        """Refresh the access token using refresh token"""
        if not self.tokens or 'refresh_token' not in self.tokens:
            raise ValueError("No refresh token available")
        
        print("🔄 Refreshing access token...")
        
        refresh_data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.tokens['refresh_token'],
            'client_id': self.client_id
        }
        
        auth = (self.username, self.password)
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.post(
                self.token_url,
                data=refresh_data,
                auth=auth,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                new_tokens = response.json()
                
                # Update tokens
                self.tokens.update(new_tokens)
                
                # Update expiration times
                now = datetime.now()
                if 'expires_in' in new_tokens:
                    self.tokens['expires_at'] = (now + timedelta(seconds=new_tokens['expires_in'])).isoformat()
                
                print("✅ Access token refreshed successfully!")
                self.save_tokens()  # Save the refreshed tokens to file
                return self.tokens
            else:
                print(f"❌ Token refresh failed: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during token refresh: {e}")
            raise
    
    def get_valid_access_token(self):
        """Get a valid access token, refreshing if necessary"""
        if not self.tokens:
            return None
        
        # Check if token is expired
        if 'expires_at' in self.tokens:
            expires_at = datetime.fromisoformat(self.tokens['expires_at'])
            if datetime.now() >= expires_at - timedelta(minutes=5):  # Refresh 5 minutes before expiry
                try:
                    self.refresh_access_token()
                except Exception as e:
                    print(f"⚠️ Failed to refresh token: {e}")
                    return None
        
        return self.tokens.get('access_token')
    
    def save_tokens(self, filename='data/sas_tokens.json'):
        """Save tokens to file"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if self.tokens:
            with open(filename, 'w') as f:
                json.dump(self.tokens, f, indent=2)
            print(f"💾 Tokens saved to {filename}")
    
    def load_tokens(self, filename='data/sas_tokens.json'):
        """Load tokens from file"""
        try:
            with open(filename, 'r') as f:
                self.tokens = json.load(f)
            print(f"📂 Tokens loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"📂 No saved tokens found at {filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading tokens: {e}")
            return False
    
    def authenticate(self, save_tokens=True):
        """Complete OAuth flow: get authorization code and exchange for tokens"""
        try:
            # Step 1: Get authorization code
            auth_code = self.get_authorization_code()
            
            # Step 2: Exchange for tokens
            tokens = self.exchange_code_for_tokens(auth_code)
            
            # Save tokens if requested
            if save_tokens:
                self.save_tokens()
            
            return tokens
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            raise
    
    def ensure_valid_token(self):
        """Ensure we have a valid access token, refreshing or re-authenticating as needed"""
        # Try to load existing tokens first
        if self.load_tokens():
            access_token = self.get_valid_access_token()
            if access_token:
                return access_token
        
        # If no valid tokens, start fresh authentication
        print("🔐 No valid tokens found. Starting authentication...")
        tokens = self.authenticate()
        return tokens['access_token']
    
    def test_token(self, access_token=None):
        """Test if an access token is valid by making a test API call"""
        if access_token is None:
            access_token = self.get_valid_access_token()
        
        if not access_token:
            return False
        
        test_endpoint = f"{self.base_url}/compute/contexts"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = requests.get(test_endpoint, headers=headers, timeout=30)
            return response.status_code in [200, 403]  # 403 means valid but limited permissions
        except requests.exceptions.RequestException:
            return False


def get_sas_tokens():
    """Get SAS tokens with automatic refresh - main entry point"""

    from config import get_config
    config = get_config()
    
    auth_client = SASViyaAuth(
        base_url=config["base_url"],
        client_id=config["client_id"],
        username=config["username"],
        password=config["password"]
    )
    
    # Try to load and refresh existing tokens first
    if auth_client.load_tokens():
        access_token = auth_client.get_valid_access_token()
        if access_token:
            return auth_client.tokens
    
    # If no valid tokens, start fresh authentication
    return auth_client.authenticate()


def create_auth_client():
    """Create and return a configured auth client"""
    try:
        from .config import get_config
    except ImportError:
        from config import get_config
    config = get_config()
    
    return SASViyaAuth(
        base_url=config["base_url"],
        client_id=config["client_id"],
        username=config["username"],
        password=config["password"]
    )


def main():
    """Main function for standalone authentication"""
    try:
        from .config import get_config, validate_env_file
    except ImportError:
        from config import get_config, validate_env_file
    
    print("🚀 SAS Viya Authentication - Smart Token Management")
    print("=" * 55)
    
    # Validate environment configuration
    if not validate_env_file():
        print("\n📋 Setup Instructions:")
        print("1. Copy env.template to .env")
        print("2. Edit .env with your actual SAS Viya credentials")
        print("3. Run the script again")
        return None
    
    config = get_config()
    
    # Create auth client
    auth_client = SASViyaAuth(
        base_url=config["base_url"],
        client_id=config["client_id"],
        username=config["username"],
        password=config["password"]
    )
    
    # Get valid access token (will auto-refresh or re-authenticate as needed)
    access_token = auth_client.ensure_valid_token()
    
    # Test the token
    print("\n🧪 Testing access token...")
    if auth_client.test_token(access_token):
        print("✅ Token test successful! You can now use the access token.")
    else:
        print("⚠️ Token test failed, but token might still work for some endpoints.")
    
    print("\n📋 Token Summary:")
    print(f"Access Token: {access_token[:50]}...")
    if 'expires_at' in auth_client.tokens:
        print(f"Expires At: {auth_client.tokens['expires_at']}")
    if 'refresh_token' in auth_client.tokens:
        print(f"Refresh Token Available: Yes")
    
    print(f"\n✅ SUCCESS! Your tokens are ready to use.")
    print(f"💡 Next time you run this, it will auto-refresh if needed!")
    
    return auth_client


if __name__ == "__main__":
    auth_client = main()
