# ************ IMPORT FRAMEWORKS ************** 
from langchain_core.documents import Document

# *************** Validator function for chek input of function as Documen Based
def validate_document_input(input_doc: any, input_name: str = "Input document") -> bool:
    """
    Validates that the provided input is a Document instance.

    Args:
        input_doc (any): The input to validate.
        input_name (str, optional): The name of the input being validated, used for error messages. Defaults to "Input document".

    Raises:
        TypeError: If the input is not a Document.

    Returns:
        bool: True if the input is a valid Document.
    """
    # *************** Check if input is an instance of Document
    if not isinstance(input_doc, Document):
        raise TypeError(f"{input_name} must be a Document instance, but got {type(input_doc)}: {input_doc}")
    
    return True

# *************** Validator function for checking if input is a list of tuples (Document, float)
def validate_context_input(context: any, input_name: str = "context") -> bool:
    """
    Validates that the provided input is a list of tuples, where each tuple contains a Document 
    and a float value.

    Args:
        context (any): The input to validate.
        input_name (str, optional): The name of the input being validated, used for error messages. Defaults to "context".

    Raises:
        TypeError: If the input is not a list.
        ValueError: If any element in the list is not a tuple with a Document and a float.

    Returns:
        bool: True if the input is a valid list of tuples (Document, float).
    """
    # *************** Check if input is a list
    if not isinstance(context, list):
        raise TypeError(f"{input_name} must be a list. but got {type(context)} : {context}")
    
    # *************** Check each item in list is a tuple (Document, float)
    for idx, item in enumerate(context):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Element {idx} of {input_name} must be a tuple with 2 elements. But got: {item}")
        
        document, similarity_value = item
        # *************** Check each document is Document based
        if not isinstance(document, Document):
            raise TypeError(f"Element {idx} of {input_name} must have a Document as the first element. But got: {type(document)}")
        # *************** Check each similarity_value is flaot
        if not isinstance(similarity_value, float):
            raise TypeError(f"Element {idx} of {input_name} must have a float as the second element. But got: {type(similarity_value)}")
    
    return True
