import os 
import logging
from dotenv import load_dotenv 
import os 

load_dotenv(override=True)

logging.basicConfig(
    filename='error.log', # Set a file for save logger output 
    level=logging.INFO, # Set the logging level
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

LOGGER = logging.getLogger(__name__)
LOGGER.info("Init Global Variable")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
ASTRADB_TOKEN_KEY = os.getenv("ASTRADB_TOKEN_KEY")
ASTRADB_API_ENDPOINT = os.getenv("ASTRADB_API_ENDPOINT")
ASTRADB_COLLECTION_NAME = os.getenv("ASTRADB_COLLECTION_NAME")
ASTRADB_KEYSPACE_NAME = os.getenv("ASTRADB_KEYSPACE_NAME")

LIST_DOC_FILE = 'LIST_DOC.json'  
DB_FILE = 'DB_FILE.json'         