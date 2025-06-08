# ***** IMPORTS *****
import re
import time
from operator                               import itemgetter

# ***** IMPORT FRAMEWORKS ***** 
from langchain.prompts                      import (
                                                PromptTemplate, 
                                                MessagesPlaceholder, 
                                                ChatPromptTemplate
                                            )
from langchain_core.messages                import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents               import Document
from langchain_core.runnables               import Runnable
from langchain_core.output_parsers          import StrOutputParser
from langchain.chains.combine_documents     import create_stuff_documents_chain

# ***** IMPORTS MODELS *****
from models.llms                            import LLMModels

# ***** IMPORTS HELPERS *****
from helpers.chat_akadbot_helpers           import (
                                                update_chat_history,
                                                get_context_based_history,
                                                get_context_based_question,
                                            )
from engine.chunk_doc                       import TokenCounter
from models.lingua  import LinguaModel

# ***** IMPORTS VALIDATORS *****
from validator.chunks_validation            import (
                                                validate_context_input,
                                                validate_document_input
                                            )
from validator.data_type_validatation       import (
                                                validate_list_input,
                                                validate_string_input
                                            )

from setup import LOGGER


def convert_chat_history(chat_history: list) -> list[BaseMessage]:
    """
    Convert chat history to the chat messages for inputted to LLM.

    Args:
        chat_history (list): List of chat messages, each containing human and AI content.

    Returns:
        list: Converted chat history with alternating HumanMessage and AIMessage objects. 
                Default to empty list if no chat_hsitory 
    """
    # ***** Return empty list because no chat_history inputted and skip the process *****
    if not chat_history:
        LOGGER.warning("No Chat History Inputted")
        return []
    
    # ***** Validate inputs chat_history is a list *****
    if not validate_list_input(chat_history, 'chat_history'):
        LOGGER.error("'chat_history' must be a list of message.")
    
    # ***** Initialize formatted history *****
    history_inputted = []

    # ***** Add messages to formatted history *****
    for chat in chat_history:
        if chat['type'] == 'human':
            history_inputted.append(HumanMessage(content=chat['content']))
        elif chat['type'] == 'ai':
            history_inputted.append(AIMessage(content=chat['content']))
    
    # ***** Log history process *****
    if history_inputted:
        LOGGER.info(f"Chat History is Converted to BaseMessages: {len(history_inputted)} messages")

    # ***** Return formatted history *****
    return history_inputted

def topic_creation(chat_history: list) -> str:
    """
    Generates a topic title summarizing the conversation in the chat history.

    This function uses a predefined prompt template to create a relevant topic title 
    based on the chat history. The generated topic follows the language and tone of 
    the conversation and is formatted in HTML.

    Args:
        chat_history (list[dict]): The chat history containing messages exchanged in the conversation.

    Returns:
        str: The generated topic title in HTML format.
    """
    # ***** Validate inputs chat_history is a list
    if not validate_list_input(chat_history, 'chat_history_topic'):
        LOGGER.error("'chat_history_topic' must be a list of message.")

    # ***** Define a template for generating the topic title *****
    topic_template = """
    Input:
        'chat_history':{chat}
    
    Instructions:
        1. Create a topic title about what the conversation is about based on the 'chat_history'.
        2. Ensure the title language and tone follow the 'chat_history'.
    
    Example Output (in HTML):
    ```html<b>Filtering Student Data for Efap Paris: Scholar Season 24-25 with Payment Confirmation</b>```
    """
    
    # ***** Initialize the prompt template *****
    topicPrompt = PromptTemplate(
        template=topic_template,
        input_variables=["chat"],
    )

    # ***** Define the processing chain for topic generation *****
    topic_chain = (
        {
            "chat": itemgetter("chat") 
        }
        | topicPrompt
        | LLMModels(temperature=1).llm_cv
    )

    # ***** Invoke the chain to generate a topic from first user question *****
    result = topic_chain.invoke(
        {
            "chat": chat_history[0] 
        }
    )

    # ***** Clean the result to remove extra characters *****
    clear_result = result.content.strip('"').replace('```html', '').replace('```', '').replace('\n', '')

    # ***** Return the cleaned topic title *****
    return clear_result

def generate_akadbot_chain() -> Runnable:
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for answering questions using an LLM.

    This function sets up a chatbot-style question-answering system where responses are 
    generated strictly based on retrieved document context. It ensures that the bot does 
    not generate answers beyond the given context.

    Returns:
        Runnable: A configured RAG chain ready for answering document-based questions.
    """

    # ***** Set QA chain prompt for bot to understand context *****
    qa_system_prompt = """
    You are an expert on the document. Generate answers only based on the given context, explain clearly. 
    Do not make up answers. If you don't know the answer based on the given context, 
    state that you can't answer questions outside of this document.
    
    The context is: '''{context}'''
    
    Please generate the answer in {language} language.
    """

    # ***** Define the chat prompt template *****
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # ***** Build RAG chain with document retriever, prompt, and LLM *****
    ragChain = create_stuff_documents_chain(
        llm=LLMModels(temperature=0.2).llm_cv,
        prompt=qa_prompt,
        output_parser=StrOutputParser(),
    )

    LOGGER.info("Akadbot Chain Generated")
    
    return ragChain

def build_reference(document: Document) -> str:
    """
    Constructs a hierarchical reference string from document metadata.

    Args:
        document (Document): A document object containing metadata.

    Returns:
        str: A formatted reference string.
    """
     # ***** Validate document input *****
    if not validate_document_input(document, 'document'):
        LOGGER.error("'document' must be a valid Document instance.")
    
    # ***** Get metadata from Document with document_name, default to "Unnamed Document" *****
    metadata = document.metadata
    reference_parts = [metadata.get("document_name", "Unnamed Document")]

    # ***** Extract hierarchical headers (header1 to header3) if available *****
    for level in range(1, 4):
        header_key = f"header{level}"
        if metadata.get(header_key):
            reference_parts.append(metadata[header_key])

    return " > ".join(reference_parts)

def get_language_used(text: str) -> str:
    """
    Get language used by sentence or text.
    Supported with Lingua language detector.

    Args:
        text (str): a string inputted need to know the language used

    Returns:
        string: a language name. (e.g. english or french)
    """
    try:
        lang_detected = LinguaModel().lingua.detect_language_of(text)
        lang_result = lang_detected.name.lower()

        LOGGER.info(f"Language used: {lang_result}")
        
        return lang_result
    
    except Exception as error_lang:
        print("An error occurred when detect langugae with Lingua:", error_lang)
        return text

def join_reference(context: list[tuple[Document, float]], header_ref: str, similarity_threshold: float = 0.6) -> str:
    """
    Joins context references where the similarity score exceeds the threshold.

    This function extracts document names and hierarchical headers from a list of documents with their
    similarity scores, ensuring unique references in a structured format.

    Args:
        context (List[Tuple[Document, float]]): A list of tuples where each contains a Document 
            object and its similarity score.
        header_ref (str): Initial reference header to be updated with extracted metadata.
        similarity_threshold (float, optional): Minimum similarity score to include a document 
            in the reference. Defaults to 0.6.

    Returns:
        str: A formatted string containing unique references extracted from the provided documents.
    """
    # ***** Input Validation *****
    if not validate_context_input(context, "context"):
        LOGGER.error("'context' Not in required format data")
    if not validate_string_input(header_ref, "header_ref", False):
        LOGGER.error("'header_ref' Not in string type")

    # ***** Initalize reference for get unique header *****
    references = set()

    # ***** Iterate through documents and extract references *****
    for document, similarity_value in context:
        if similarity_value > similarity_threshold:
            reference = build_reference(document)
            references.add(reference)
        
    # ***** Remove duplicate references *****
    unique_lines = set(header_ref.split('\n'))
    header_ref = '\n'.join(unique_lines)

    return header_ref

def rag_ask_stream(question: str, 
                          course_id: str, 
                          chat_history: list = [], 
                          topic: str = ''):
    """
    Streaming version of ask_with_memory function using LangChain streaming.
    
    This function yields streaming responses from the RAG chain while maintaining
    all the original functionality for context retrieval and chat history management.
    
    Args:
        question (str): The question asked by the student.
        course_id (str): The identifier of the course for context retrieval.
        chat_history (optional, list[dict]): The previous chat conversation to maintain context.
        topic (optional, str): The current topic of discussion. If empty, a topic is generated.
    
    Yields:
        dict: Streaming chunks containing partial responses and metadata
        
    Returns:
        dict: Final response with complete message and metadata
    """
    
    # ***** Input Validation (same as original) *****
    if not validate_string_input(question, "question"):
        LOGGER.error("'question' must be a string and not empty")
    if not validate_string_input(course_id, "course_id"):
        LOGGER.error("'course_id' must be a string and not empty")
    if chat_history and not validate_list_input(chat_history, "chat_history", False):
        LOGGER.error("'chat_history' must be a list and not empty")
    if topic and not validate_string_input(topic, "topic", False):
        LOGGER.error("'topic' must be a string and not empty")

    # ***** Initialize variables *****
    message_chunks = []
    full_message = ''
    header_ref = ''
    
    # ***** Detect language used from current question *****
    lang_used = get_language_used(question)

    # ***** Setup token count *****
    count_token = TokenCounter()
    
    start_time = time.time()

    # ***** Generate Akadbot RAG Chain (streaming enabled) *****
    ragChain = generate_akadbot_chain()

    # ***** Convert chat history into structured format
    history_input = convert_chat_history(chat_history)
    history_input.extend([HumanMessage(content=question)])

    # ***** Retrieve relevant context from past conversations or documents *****
    if chat_history:
        context = get_context_based_history(history_input, course_id)
    else:
        context = get_context_based_question(question, course_id)

    LOGGER.info(f"CONTEXT: {len(context)}\n{context}")

    # ***** Get document chunks from context *****
    docs = [doc[0] for doc in context]

    # ***** Stream the response using the RAG chain *****
    try:
        for chunk in ragChain.stream({
            "context": docs,
            "messages": history_input,
            "language": lang_used,
        }):
            # ***** Extract content from the streaming chunk *****
            chunk_content = chunk if isinstance(chunk, str) else str(chunk)
            message_chunks.append(chunk_content)
            full_message += chunk_content
            
            # ***** Yield streaming response *****
            yield {
                "type": "stream",
                "chunk": chunk_content,
                "partial_message": full_message,
                "is_complete": False
            }
    
    except Exception as e:
        LOGGER.error(f"Streaming error: {e}")
        # ***** Fallback to non-streaming if streaming fails *****
        full_message = ragChain.invoke({
            "context": docs,
            "messages": history_input,
            "language": lang_used,
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    LOGGER.info(f"TIME TO INVOKE: {elapsed_time} seconds")

    # ***** Compile reference headers from retrieved context *****
    header_ref = join_reference(context, header_ref)

    # ***** Add current question and answer into chat history with reference *****
    update_chat_history(chat_history, question, full_message, header_ref)

    # ***** Check if topic exists; if not, generate a topic summary *****
    if topic == '':
        LOGGER.info("No topics inputted")
        topic = topic_creation(chat_history)

    # ***** Get token cost track *****
    last_human_message = history_input[-1].content 
    tokens_in = count_token.count_tokens(last_human_message)
    tokens_out = count_token.count_tokens(full_message)
    
    # ***** Final response *****
    final_response = {
        "type": "complete",
        "message": full_message,
        "topic": topic,
        "chat_history": chat_history,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "is_complete": True
    }
    
    yield final_response
    return final_response


if __name__ == "__main__":
    pass