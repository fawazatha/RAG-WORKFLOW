from setup                              import LOGGER

# ***** IMPORTS FRAMEWORKS *****
from langchain_core.messages            import HumanMessage, BaseMessage
from langchain_core.documents           import Document

# ***** IMPORTS HELPERS *****
from helpers.astradb_connect_helper     import get_vector_collection

# ***** IMPORTS VALIDATORS *****
from validator.data_type_validatation   import (
                                            validate_string_input,
                                            validate_list_input,
                                            validate_int_input
                                        )

# ***** Function to retrieve context-based question with vector simlarity
def get_context_based_question(query: str, course_id: str) -> list[tuple[Document, float]]:
    """
    Retrieves relevant documents based on the given question and course ID.

    Performs a similarity search within the vector database to find 
    the most relevant documents related to the given query with a relevance score above the threshold (0.4) 
    are returned.

    Args:
        query (str): The student's question to search for relevant context.
        course_id (str): The unique identifier of the course to filter relevant documents.

    Returns:
        list[tuple[Document, float]]: A list of tuples containing:
            - Document: The retrieved document relevant to the query.
            - float: The relevance score of the document.
    """
    # ***** Validate inputs query is not an empty-string
    if not validate_string_input(query, 'query'):
        LOGGER.error("'query' must be a string.")
    # ***** Validate inputs course_id is not an empty-string
    if not validate_string_input(course_id, 'course_id'):
        LOGGER.error("'course_id' must be a string.")

    # ***** Retrieve vector collection for similarity search
    vector_coll = get_vector_collection()

    # ***** Perform similarity search with relevance filtering
    relevant_docs_score = vector_coll.similarity_search_with_relevance_scores(
        query=query, k=4, filter={'course_id': course_id}, score_threshold=0.4
    )

    return relevant_docs_score

# ***** Function to retrieve context-based history from conversation
def get_context_based_history(conversation: list[BaseMessage], course_id: str) -> list[tuple[Document, float]]:
    """
    Retrieves relevant contextual history based on recent human messages in the conversation.

    This function extracts the latest user questions from the conversation, retrieves context 
    related to each question, and compiles a history of relevant context.

    Args:
        conversation (list[BaseMessage]): A list of message objects representing the conversation.
        course_id (str): The identifier for the course to filter context.

    Returns:
        list[tuple[Document, float]]: A list of context data associated with the retrieved questions.
    """
    # ***** Validate inputs conversation is a list
    if not validate_list_input(conversation, 'conversation'):
        LOGGER.error("'conversation' must be a list of message.")
        # ***** Validate inputs course_id is not an empty-string
    if not validate_string_input(course_id, 'course_id'):
        LOGGER.error("'course_id' must be a string.")
        
    # ***** Extract latest user questions
    conversation_question = get_question_history(conversation)

    context_history = []

    # ***** Retrieve context for each question
    for question in conversation_question:
        question_context = get_context_based_question(question, course_id)
        for context in question_context:
            context_history.extend([context])

    return context_history

# ***** Function to get recent human questions from conversation
def get_question_history(conversation: list[BaseMessage], latest_chat: int = 2) -> list[HumanMessage]:
    """
    Extracts the latest human messages from a conversation.

    This function retrieves a specified number of recent human messages from a conversation 
    history, filtering only messages sent by the human user.

    Args:
        conversation (list[BaseMessage]): A list of message objects representing the conversation.
        latest_chat (int, optional): The number of latest exchanges to retrieve. Defaults to 2.

    Returns:
        list[HumanMessage]: A list of recent human messages from the conversation.
    """

    # ***** Validate inputs conversation is a list
    if not validate_list_input(conversation, 'conversation'):
        LOGGER.error("'conversation' must be a list of message.")
    # ***** Validate inputs conversation is interger
    if latest_chat and not validate_int_input(latest_chat, 'latest_chat'):
        LOGGER.error("'latest_chat' must be an interger.")
    
    # ***** Determine the starting index based on latest_chat
    # ***** Adjusting to retrieve both user and assistant messages
    length = -(latest_chat * 2)
    question = []

    # ***** Extract human messages from conversation
    for chat in conversation[length:]:
        if chat.type == 'human':
            question.append(chat.content)

    return question

# ***** Function helper to extend latest chat_history with new response
def update_chat_history(chat_history: list[dict], question: str, answer: str, reference: str = '') -> None:
    """
    Used for extend latest chat_hsitory with user request question and response of akadbot. 


    Args:
        chat_history (list[dict]): list of dict latest chat history inputted to akadbot.
        question (str): string of user input question to document.
        answer (str): string of response by akadbot.
        reference (str): string of metadata header from context. saved to chat_history with type "ai"

    Returns:
        None
    """
    # ************* Validate chat_history in list
    if not validate_list_input(chat_history, "chat_history", False):
        LOGGER.error("chat_history must be a non-empty list of dictionaries.")
    # ***** Validate input question not
    if not validate_string_input(question, 'question_history'):
        LOGGER.error("'question' must be a non-empty string.")
    # ***** Validate input answer not
    if not validate_string_input(answer, 'answer_history'):
        LOGGER.error("'answer' must be a non-empty string.")
    # ***** Validate input reference not
    if reference and not validate_string_input(reference, 'reference_history'):
        LOGGER.error("'reference' must be a non-empty string.")

    # ***** add question and answer to each dict type to last index with extend
    chat_history.extend(
                [
                    {"type": "human", "content": question}, 
                    {"type": "ai", "content": answer, 'header_ref': reference}
                ]
            )
