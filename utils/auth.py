import base64
import requests
import streamlit as st

def get_access_token():
    """Get Spotify access token using client credentials flow"""
    try:
        client_id = st.secrets["CLIENT_ID"]
        client_secret = st.secrets["CLIENT_SECRET"]
    except KeyError as e:
        st.error(f"Missing Spotify credential: {e}")
        st.error("Please add CLIENT_ID and CLIENT_SECRET to your Streamlit secrets")
        return None
    
    auth_url = 'https://accounts.spotify.com/api/token'

    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode('utf-8')
    headers = {
        'Authorization': f'Basic {auth_header}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}
    
    try:
        response = requests.post(auth_url, headers=headers, data=data, timeout=10)
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['access_token']
        elif response.status_code == 400:
            st.error("❌ Invalid Spotify credentials. Please check your CLIENT_ID and CLIENT_SECRET.")
            return None
        else:
            st.error(f"❌ Authentication failed with status code: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("❌ Authentication request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Spotify authentication server. Check your internet connection.")
        return None
    except ValueError:
        st.error("❌ Failed to parse authentication response.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected authentication error: {str(e)}")
        return None