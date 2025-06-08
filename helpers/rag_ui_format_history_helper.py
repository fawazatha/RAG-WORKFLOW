# ********** function format chat history to pass in function ask_to_memory
def format_chat_history(chat_history: list) -> list:
    """
    Formats a chat history list into a standard dictionary format.

    Args:
        chat_history (list): A list of messages, where each message is either:
            - An object with attributes `type`, `content`, and optionally `header_ref`.
            - A dictionary with keys "type", "content", and optionally "header_ref".

    Returns:
        list: A list of formatted dictionaries with the keys:
            - `type` (str): The type of the message (e.g., 'ai', 'human').
            - `content` (str): The content of the message.
            - `header_ref` (str): The header reference for AI messages (optional, default to an empty string).
    
    """
    formatted_chat_history = []

    for msg in chat_history:
        if isinstance(msg, dict):
            # ********** Extract from dictionary
            msg_type = msg.get("type")
            msg_content = msg.get("content")
            msg_header_ref = msg.get("header_ref", "") if msg_type == "ai" else ""
        elif hasattr(msg, 'type') and hasattr(msg, 'content'):
            # ********** Extract from object
            msg_type = getattr(msg, 'type', None)
            msg_content = getattr(msg, 'content', None)
            msg_header_ref = getattr(msg, 'header_ref', "") if msg_type == "ai" else ""
        else:
            raise ValueError(f"Invalid message format: {msg}")

        # ********** Validate required fields
        if not msg_type or not msg_content:
            raise ValueError(f"Message is missing required fields: {msg}")

        # ********** Append the formatted dictionary
        formatted_chat_history.append({
            "type": msg_type,
            "content": msg_content,
            "header_ref": msg_header_ref
        })

    return formatted_chat_history

# ********** format chat history from history returned function ask_to_memory
def format_and_extract_header_returned(chat_history: list) -> tuple:
    """
    Formats a chat history list and extracts the `header_ref` for the last AI message.

    Args:
        chat_history (list): A list of messages, where each message is either:
            - An object with attributes `type`, `content`, and optionally `header_ref`.
            - A dictionary with keys "type", "content", and optionally "header_ref".

    Returns:
        tuple: A tuple containing:
            - list: A list of formatted dictionaries with keys:
                - `type` (str): The type of the message (e.g., 'ai', 'human').
                - `content` (str): The content of the message.
                - `header_ref` (str): The header reference for AI messages (optional, default to an empty string).
            - str: The `header_ref` of the last AI message, or an empty string if no AI messages are present.

    """
    formatted_returned_history = []
    header_ref_extracted = ""

    for msg in chat_history:
        if not isinstance(msg, dict):
            raise ValueError(f"Invalid message format: {msg}")
        if isinstance(msg, dict):
            # ********** Process as a dictionary
            msg_dict = {
                "type": msg.get("type"),
                "content": msg.get("content"),
            }
            if msg_dict["type"] == "ai":
                msg_dict["header_ref"] = msg.get("header_ref", "")
                header_ref_extracted = msg.get("header_ref", "")
            formatted_returned_history.append(msg_dict)
        elif hasattr(msg, 'type') and hasattr(msg, 'content'):
            # ********** Process as an object
            msg_dict = {
                "type": getattr(msg, 'type', None),
                "content": getattr(msg, 'content', None),
            }
            if msg_dict["type"] == "ai":
                msg_dict["header_ref"] = getattr(msg, 'header_ref', "")
                header_ref_extracted = getattr(msg, 'header_ref', "")
            formatted_returned_history.append(msg_dict)
        # else:
        #     raise ValueError(f"Invalid message format: {msg}")

    return formatted_returned_history, header_ref_extracted
