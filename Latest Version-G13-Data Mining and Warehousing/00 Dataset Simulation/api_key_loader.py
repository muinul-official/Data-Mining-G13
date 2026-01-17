"""
Secure API Key Loading Utility
Use this code in all notebooks to load API keys securely
"""

import os
import sys

def load_api_key():
    """
    Securely load Google API key from environment.
    
    Works in both Google Colab and local environments.
    
    Returns:
        bool: True if API key loaded successfully, False otherwise
    """
    try:
        # Try Google Colab's secure secret management first
        from google.colab import userdata
        os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
        print("✓ API key loaded from Colab secrets")
        return True
    except ImportError:
        # Not in Colab, try loading from .env file for local development
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            if os.getenv("GOOGLE_API_KEY"):
                print("✓ API key loaded from .env file")
                return True
            else:
                print("⚠ WARNING: GOOGLE_API_KEY not found in .env file!")
                print("Please create a .env file with your API key.")
                print("See .env.example for the template.")
                return False
        except ImportError:
            print("⚠ WARNING: python-dotenv not installed!")
            print("Install it with: pip install python-dotenv")
            return False
    except Exception as e:
        print(f"❌ ERROR loading API key: {e}")
        return False


# Usage in notebooks:
# Replace the hardcoded API key section with:
#
# from utils.api_key_loader import load_api_key
# load_api_key()
#
# OR copy the load_api_key() function directly into your notebook cell
