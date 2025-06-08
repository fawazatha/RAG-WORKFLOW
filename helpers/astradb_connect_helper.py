
from langchain_astradb.vectorstores     import AstraDBVectorStore
from astrapy                            import DataAPIClient, Collection
from models.embeddings import EmbeddingsModels
# *************** LOAD ENVIRONMENT ***************
from setup                              import (
                                            ASTRADB_TOKEN_KEY,
                                            ASTRADB_API_ENDPOINT,
                                            ASTRADB_COLLECTION_NAME,
                                            ASTRADB_KEYSPACE_NAME
                                        )

def get_vector_collection() -> AstraDBVectorStore:
    """
    Initialize the AstraDBVectorStore and get the AstraDB vector of document collection.
    This function become RAG engine for get context with as_retriever()

    Returns:
        AstraDBVectorStore (object): The AstraDB vector collection object.
    """

    try:
        vector_store_integrated = AstraDBVectorStore(
            collection_name=ASTRADB_COLLECTION_NAME,
            api_endpoint=ASTRADB_API_ENDPOINT,
            token=ASTRADB_TOKEN_KEY,
            namespace=ASTRADB_KEYSPACE_NAME,
            autodetect_collection=False,
            embedding=EmbeddingsModels().embedding_large_gemini, 
            content_field="page_content"
        )
        return vector_store_integrated
    except Exception as error:
        # ****** Handle exceptions ******
        raise ConnectionError(f"Failed to connect to AstraDB {ASTRADB_COLLECTION_NAME}: {str(error)}")

def get_astradb_doc_chunked():
    """
    Initialize the DataAPIClient and get the AstraDB doc_chunked collection.

    Returns:
        astrapy_collection: The AstraDB collection object.
    """
    # Initialize the DataAPI client globally
    try:
        client = DataAPIClient(ASTRADB_TOKEN_KEY)
        database = client.get_database(ASTRADB_API_ENDPOINT)
        collection = database.get_collection(
            ASTRADB_COLLECTION_NAME, 
            keyspace=ASTRADB_KEYSPACE_NAME)
        return collection    
    except Exception as error:
        # ****** Handle exceptions ******
        raise ConnectionError(f"Failed to connect to AstraDB: {str(error)}")
