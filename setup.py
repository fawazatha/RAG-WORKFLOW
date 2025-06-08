import os 
import logging
from dotenv import load_dotenv 
import os 
import streamlit as st 

api_key = st.secrets["api"]
load_dotenv(override=True)

logging.basicConfig(
    filename='error.log', # Set a file for save logger output 
    level=logging.INFO, # Set the logging level
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

LOGGER = logging.getLogger(__name__)
LOGGER.info("Init Global Variable")

GOOGLE_API_KEY = api_key['GOOGLE_API_KEY']
ASTRADB_TOKEN_KEY = api_key["ASTRADB_TOKEN_KEY"]
ASTRADB_API_ENDPOINT = api_key["ASTRADB_API_ENDPOINT"]
ASTRADB_COLLECTION_NAME = api_key["ASTRADB_COLLECTION_NAME"]
ASTRADB_KEYSPACE_NAME = api_key["ASTRADB_KEYSPACE_NAME"]

DB_FILE = api_key['DATABASE_CHAT_TOPIC']
LIST_DOC_FILE = api_key["DATA_LIST_DOC"]    