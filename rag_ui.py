# ******* IMPORTS FRAMEWORK *******
import streamlit            as st
from streamlit_option_menu  import option_menu

# ******* IMPORTS *******
import json
import os
from datetime               import datetime
import threading
import uuid 
from astrapy                import DataAPIClient
import time 
from typing import List 

from engine.chunk_doc import DocProcess 
from engine.rag_chat import rag_ask_stream

# ******* IMPORTS SETUP *******
from setup                                      import (ASTRADB_TOKEN_KEY, 
                                                        ASTRADB_API_ENDPOINT, 
                                                        LIST_DOC_FILE,
                                                        DB_FILE, 
                                                        ASTRADB_COLLECTION_NAME,
                                                        LOGGER)
# ******* IMPORTS HELPER *******
from helpers.rag_ui_styling_helper           import (styling, plot_title, 
                                                        padding_height, 
                                                        plot_title_upload_doc, 
                                                        plot_title_select_document,
                                                        plot_title_chat_topic)
from helpers.rag_ui_format_history_helper    import (format_chat_history, 
                                                        format_and_extract_header_returned)
# ******* IMPORTS VALIDATOR *******
from validator.data_type_validatation           import (validate_dict_input,  
                                                        validate_list_input, 
                                                        validate_string_input)



st.set_page_config(layout="wide")

# ******* padding height
padding_height()

# ******* list docs json
with open(LIST_DOC_FILE, "r") as file:
    LIST_DOCS = json.load(file)

def save_list_to_json(list_docs: list) -> None:
    """
    Saves a list of dictionaries to a JSON file.

    Args:
        list_docs (list): The list of dictionaries to save.
        file_name (str): The name of the JSON file to save to.

    Returns:
        None
    """
    if not validate_list_input(list_docs, "list_docs"):
        raise ValueError("list_docs must be a list of dictionaries.")
    
    with open('LIST_DOC.json', "w") as json_file:
        json.dump(list_docs, json_file, indent=4)

# ******* initiate json list dict
def initialize_db():
    """
    Initiate json to save topic chat
    """
    
    # ******* initiate default format json
    default_db = [
        {
            "topic": "New Chat",
            "message": [],
            "created_at": "",
            "document": ""
        }
    ]
    
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as file:
            json.dump(default_db, file, indent=2)
    else:
        # ******* Check if file is empty or invalid
        try:
            with open(DB_FILE, 'r') as file:
                json.load(file)
        except json.JSONDecodeError:
            # ******* If file is empty or invalid, write default structure
            with open(DB_FILE, 'w') as file:
                json.dump(default_db, file, indent=2)

# ******* load chat history
def load_chat_history():
    """
    function to load chat history
    """
    
    try:
        with open(DB_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # ******* If error loading the file, initialize with empty list
        default_db = []
        with open(DB_FILE, 'w') as file:
            json.dump(default_db, file, indent=2)
        return default_db

# ******* save chat history to json
def save_chat_history(topics_list):
    """
    function to save chat history
    """
    
    # ******* validate topic_list 
    if not validate_list_input(topics_list, "topic_list", False):
        LOGGER.error("topic_list must be a list")
    
    try:
        with open(DB_FILE, 'w') as file:
            json.dump(topics_list, file, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

# ******* get source doc by course name
def get_source_by_name(doc_name: str) -> str:
    """
    Function to get course_id by its name
    
    Args:
        doc_name: document name  

    Returns:
        course_id:  course id 
    """
    
    # ******* validate doc name data type 
    if not validate_string_input(doc_name, "doc_name"):
        LOGGER.error("doc_name must be a string")
            
    for doc in LIST_DOCS:
        if doc["course_name"] == doc_name:
            return doc["course_id"]
    return None

# ******* function to format topic creation
def format_topic_name(topic: str) -> str:
    """
    Function to format topic name to replace <br> with empty string
    Args:
        topic (str): topic created

    Returns:
        topic (str): topic created
    """
    
    # ******* validate topic 
    if not validate_dict_input(topic, "topic"):
        LOGGER.error(f"{topic} must be a dict")
    
    # ******* format to replace <br>
    topic_name = topic.get('topic', 'Unnamed Topic').replace("<b>", "").replace("</b>", "")
    created_at = topic.get('created_at', '')
    
    if created_at:
        return f"{topic_name}"
    return topic_name

# ******* function to handle buble chat to clear and create new default topic
def clear_current_chat(db_data: List[dict]):
    """
    function to handle buble chat when click start new chat 
    
    Args:
        db_data (_type_): _description_
    """
    
    # ******* initiate default topic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_topic = f"New Chat_{timestamp}"

    # ******* Save current messages into history if not already saved
    if st.session_state.get("messages", []):
        current_topic = st.session_state.get("current_topic", "")
        if current_topic.startswith("New Chat_") or len(st.session_state["messages"]) == 0:
            LOGGER.info("New Chat topic is empty and will not be saved.")
        elif any(entry["topic"] == current_topic for entry in db_data):
            LOGGER.info("Current topic already saved in DB, skipping save.")
        else:
            db_data.append({
                "topic": current_topic,
                "messages": st.session_state["messages"],
                "created_at": datetime.now().isoformat(),
                "document": st.session_state.get("selected_doc", ""),
            })
            LOGGER.info("Saved current topic to DB.")

    # ******* Reset session state for new chat
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.current_topic = new_topic

    # ******* Avoid duplicate new topics in db_data
    existing_topics = [entry["topic"] for entry in db_data]
    if new_topic not in existing_topics:
        db_data.append({
            "topic": new_topic,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "document": st.session_state.get("selected_doc", ""),
        })
        LOGGER.info(f"Added new topic: {new_topic}")
    else:
        LOGGER.info(f"Topic {new_topic} already exists in DB.")

    # ******* Save changes to database
    save_chat_history(db_data)

    st.sidebar.success("Started a new chat while preserving history!")
    st.rerun()

# ******* function to wait courseid to enter to astradb
def waiting_course_time(course_id: str, timeout=180) -> bool:
    """
    Function to wait course id enetered to database
    
    Args:
        course_id (_type_): course id generated with uuid
        timeout (int, optional): Defaults to 180.

    """
    # ******* initiate time and astradb client
    start_time = time.time()
    client = DataAPIClient(ASTRADB_TOKEN_KEY)
    database = client.get_database(ASTRADB_API_ENDPOINT)
    coll = database.get_collection(ASTRADB_COLLECTION_NAME)
    
    # ******* loop based time to find course id
    while time.time() - start_time < timeout:
        course_exists = coll.find(
                                filter = {'metadata.course_id': course_id}
                            )
        
        # ******* Wait for 5 seconds before checking again
        time.sleep(5)
        if course_exists:
            st.session_state["waiting_for_course"] = True
            return True
       
    st.session_state["waiting_for_course"] = False
    return False

# ******* function for on-click callback when upload doc
def upload_doc_button(data: dict):
    """
    Callback for button to upload document 
    
    Args:
        data (dict): data payload dict with no doc_id
    """
    doc_processing = DocProcess()
    # ******* validate data input 
    if not validate_dict_input(data, "dict"):
        LOGGER.error(f"data input must be a dict")
    
    with st.sidebar:
        # ******* validate input data present
        if not data.get("doc_url")  or not data.get("doc_name") or not data.get("course_name"):
            LOGGER.error("Fill in all fields before proceeding.")
            return
        
        if not data.get("course_id") or not data.get("document_id"):
            LOGGER.error("course_id or document_id is not generated by uuid.")
            return
       
        # ******* session for condition button dissable
        st.session_state["waiting_for_course"] = True

        # ******* procces document upload
        processPDFThread = threading.Thread(
                target=doc_processing.process_documents,
                args=(data.get("doc_url"), 
                    data.get("course_id"), 
                    data.get("doc_name"), 
                    data.get("course_name"), 
                    data.get("document_id"))
            )
        processPDFThread.start()
        
# ******* main function to run akabot streamlit ui
def rag_ui():
    """
    Main function to render the Akabot Streamlit UI.
    
    Handles interactions for the Akabot application.
    """
    # ***** initiate db
    initialize_db()
    # ***** styling
    styling()
    # ***** Load existing chat history
    db_data = load_chat_history()
    # ***** initiate timestamp for default current topic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ***** Initialize session states with proper message structure
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = f"New Chat_{timestamp}"
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = False 

    if "waiting_for_course" not in st.session_state:
        st.session_state.waiting_for_course = False  
        
    if "course_id" not in st.session_state:
        unique_id = uuid.uuid4()
        st.session_state["course_id"] = f"course_{unique_id}"
        st.session_state["document_id"] = f"document_{unique_id}"
    
    # ***** ui for title in main display 
    text = plot_title()
    st.markdown(text, unsafe_allow_html=True)
    st.caption("Talk with your document")
    
    # ***** Sidebar for upload and buble chat
    with st.sidebar:
        # ***** plot title upload doc in sidebar
        plot_title_upload_doc()
        
        # ***** Input fields for upload doc with callback
        doc_url = st.text_input("Document URL")
        doc_name = st.text_input("Document Name")
        course_name = st.text_input("Course Name")
        
        # ***** button upload doc with callback
        if st.button(
            "Upload",
            key="upload_doc",
            on_click= lambda data={
                "doc_url":doc_url, 
                "doc_name":doc_name, 
                "course_name":course_name, 
                "course_id": st.session_state["course_id"], 
                "document_id": st.session_state["document_id"]
                } :upload_doc_button(data),
            disabled= st.session_state["waiting_for_course"]
        ):

            # ***** Wait for course ID
            with st.spinner("Waiting for course ID to be ready in the database..."):
                    course_id_ready = waiting_course_time(st.session_state["course_id"])

            # ***** shows message and append session state
            if course_id_ready:
                    st.success("Course ID successfully entered in the database.")
                    # ***** Append to session state for list doc once course_id ready uploaded
                    LIST_DOCS.append({
                        "course_id": st.session_state["course_id"],
                        "course_name": course_name
                    })
                    save_list_to_json(LIST_DOCS)
            else:
                    st.error("Timeout: Course ID was not entered in the database within 3 minutes.")

            # ***** Reset state after processing
            st.session_state["waiting_for_course"] = False

        # ***** add space
        for _ in range(1):
            st.write("")
        
        # ***** plot title select document in sidebar 
        plot_title_select_document()

        # ***** select box to choose course_name based on list_docs
        selected_doc = st.selectbox(
            "",
            options=[doc["course_name"] for doc in LIST_DOCS],
            index=0
        )
       
        # ***** add space
        for _ in range(1):
            st.write("")
        
        # ***** plot title chat topic in sidebar
        plot_title_chat_topic()

        # ***** sorted topic based on created_at date
        topics = sorted(db_data, key=lambda x: x.get('created_at', ''), reverse=True)

        # ***** handle if topics to be buble chat using option_menu
        if topics:
            selected_topic = option_menu(
                menu_title="",
                options=[format_topic_name(topic) for topic in topics],
                icons=["chat-dots"] * len(topics),
                menu_icon=None,
                default_index=0 if topics else None,
                styles={
                    "nav-link-selected": {"background-color": "#10B981"},
                    "nav-link": {"white-space": "normal", "height": "auto", "min-height": "44px"}
                }
            )
            
            selected_base_topic = selected_topic #.split(" (")[0] if selected_topic else None
            
            # ***** if selected topic normalized json to replace <br>
            if selected_base_topic:
                normalized_db_data = [
                    {
                        **item,
                        "normalized_topic": item["topic"].replace("<b>", "").replace("</b>", "")
                    }
                    for item in db_data
                ]
                
                topic_data = next(
                    (item for item in normalized_db_data if item['normalized_topic'] == selected_base_topic), 
                    None
                )
                            
                if topic_data:
                    # ***** Check if the key is 'message' or 'messages' 
                    if 'message' in topic_data:
                        st.session_state.messages = topic_data['message']  
                    elif 'messages' in topic_data:
                        st.session_state.messages = topic_data['messages']  
                    else:
                        st.session_state.messages = []  
                    
                    # ***** Update current topic
                    st.session_state.current_topic = topic_data['topic']

            # ***** handle button start new chat
            clear_button = st.button("Start New Chat", key="clear_button") 
            if clear_button:
                clear_current_chat(db_data)
            
            # ***** tooltip for clarity  
            st.markdown("""
                <div style="margin: 10px 0; border-bottom: 1px solid #ccc;"></div>
                <div style="font-size: 0.8em; color: #666;">
                    Click 'Start New Chat' to begin a new topic doc.
                </div>
            """, unsafe_allow_html=True)
            
    # ***** Main chat interface
    source = get_source_by_name(selected_doc)

    # ***** Chat container
    chat_container = st.container(border=True, height=600)

    # ***** Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # ***** Add user message to session state first
        st.session_state.messages.append({"type": "human", "content": prompt})
        
        # ***** Now display everything in the chat container
        with chat_container:
            # ***** Display all previous messages
            for message in st.session_state.messages:
                if isinstance(message, dict) and "type" in message and "content" in message:
                    with st.chat_message(message["type"]):
                        st.markdown(message["content"])
                        if message["type"] == "ai" and "header_ref" in message and message["header_ref"]:
                            st.markdown(
                                f"""
                                <p style="color: gray; font-size: 12px;">
                                    References: {message['header_ref'].replace("- ", "<br>-")}
                                </p>
                                """,
                                unsafe_allow_html=True
                            )
            
            # ***** Stream the new AI response
            with st.chat_message("ai"):
                message_placeholder = st.empty()
                full_response = ""
                header_ref = ""
                
                formatted_chat_history = format_chat_history(st.session_state.chat_history)
                current_topic = "" if st.session_state.current_topic.startswith("New Chat") else st.session_state.current_topic
                
                # ***** Stream the response
                for response in rag_ask_stream(prompt, source, formatted_chat_history, current_topic):
                    if response["type"] == "stream":
                        full_response = response["partial_message"]
                        message_placeholder.markdown(full_response + "▌")
                        
                    elif response["type"] == "complete":
                        full_response = response["message"]
                        chat_history = response["chat_history"]
                        topics = response["topic"]
                        
                        # ***** Show final message without cursor
                        message_placeholder.markdown(full_response)
                        
                        # ***** Process and show references
                        formatted_returned_history, header_ref_extracted = format_and_extract_header_returned(chat_history)
                        header_ref = formatted_returned_history[-1].get("header_ref", header_ref_extracted)
                        
                        if header_ref:
                            st.markdown(
                                f"""
                                <p style="color: gray; font-size: 12px;">
                                    References: {header_ref.replace("- ", "<br>-")}
                                </p>
                                """,
                                unsafe_allow_html=True
                            )
        
        # ***** Update session state after streaming
        st.session_state.chat_history = formatted_returned_history
        st.session_state.current_topic = topics
        
        # ***** Add AI message to session state
        ai_message = {
            "type": "ai",
            "content": full_response,
            "header_ref": header_ref
        }
        st.session_state.messages.append(ai_message)
        
        # ***** Update database (same as before)
        if topics:  
            db_data = load_chat_history()
            db_data = [entry for entry in db_data if not entry["topic"].startswith("New Chat")]
            topic_found = False 
            
            for entry in db_data:
                if entry["topic"] == topics:
                    entry["message"] = st.session_state.messages
                    entry["created_at"] = datetime.now().isoformat()
                    entry["document"] = selected_doc
                    topic_found = True
                    break
            
            if not topic_found:
                db_data.append({
                    "topic": topics,
                    "message": st.session_state.messages,
                    "created_at": datetime.now().isoformat(),
                    "document": selected_doc
                })
            
            save_chat_history(db_data)
        
        # ***** Rerun to refresh the display
        st.rerun()

    # ***** If no new input, just display existing messages in container
    else:
        with chat_container:
            for message in st.session_state.messages:
                if isinstance(message, dict) and "type" in message and "content" in message:
                    with st.chat_message(message["type"]):
                        st.markdown(message["content"])
                        if message["type"] == "ai" and "header_ref" in message and message["header_ref"]:
                            st.markdown(
                                f"""
                                <p style="color: gray; font-size: 12px;">
                                    References: {message['header_ref'].replace("- ", "<br>-")}
                                </p>
                                """,
                                unsafe_allow_html=True
                            )
                            
if __name__ == '__main__':
    rag_ui()