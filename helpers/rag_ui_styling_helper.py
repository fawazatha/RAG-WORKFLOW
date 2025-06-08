# ************ IMPORT FRAMEWORK ************
import streamlit as st 

# ************ styling streamlit for textinput, button, selectbox
def styling():
    st.markdown("""
    <style>
     /* Style for the text input field  TEXT INPUT*/
    .stTextInput input {
        background-color: #ffffff;        /* White background */
        color: #000000;                   /* Black text color */
        padding: 10px 15px;               /* Padding for a spacious look */
        font-size: 16px;                  /* Font size */
        border: 1px solid #000000;        /* Thin black border */
        border-radius: 7px;              /* Rounded corners for smooth look */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        transition: box-shadow 0.3s ease; /* Smooth shadow transition */
    }

    /* Hover effect for the text input */
    .stTextInput input:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Enhanced shadow on hover */
    }

    /* Focus effect for the text input */
    .stTextInput input:focus {
        outline: none;                    /* Remove default focus outline */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); /* Stronger shadow on focus */
    }
    
    /* styling button */
    .stButton > button {
                        background-color: #10B981;
                        border: none;
                        color: white;
                        padding: 8px 16px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 12px;
                    }
    
    /* Style for the selectbox component */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff;       /* White background */
        color: #000000;                  /* Black text color */
        padding: 5px 6px;              /* Padding for a spacious look */
        font-size: 16px;                 /* Font size */
        border: 1px solid #000000;       /* Thin black border */
        border-radius: 7px;              /* Rounded corners for a smooth look */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        transition: box-shadow 0.3s ease; /* Smooth shadow transition */
    }

    /* Hover effect for the selectbox */
    .stSelectbox div[data-baseweb="select"]:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Enhanced shadow on hover */
    }
    </style>
    """, unsafe_allow_html=True)

# ************ styling title in main display
def plot_title():
    gradient_text_html = """
    <style>
    .gradient-text {
        font-weight: bold;
        background: linear-gradient(to right, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-size: 3em;
    }
    </style>
    <div class="gradient-text">RAG WORKFLOW</div>
    """
    return gradient_text_html

# ************ styling remove height padding
def padding_height():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# ************ styling streamlit title upload doc in sidebar
def plot_title_upload_doc():   
    st.markdown("""
                <div style="
                    background-color: #ffffff;
                    border-radius: 5px;
                    padding: 3px;
                    margin-bottom: 3px;
                    text-align: center;
                    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
                ">
                    <h3 style="
                        color: #333333;
                        font-size: 12px;
                        font-weight: bold;
                        margin: 0;
                        text-transform: uppercase;
                        letter-spacing: 1.2px;
                    ">Upload Document</h3>
                </div>
            """, unsafe_allow_html=True)

# ************ styling streamlit title select doc in sidebar
def plot_title_select_document():
    st.markdown("""
            <div style="
                background-color: #ffffff;
                border-radius: 5px;
                padding: 3px;
                margin-bottom: 3px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            ">
                <h3 style="
                    color: #333333;
                    font-size: 12px;
                    font-weight: bold;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                ">Select Document</h3>
            </div>
        """, unsafe_allow_html=True)

# ************ styling streamlit title chat topic in sidebar
def plot_title_chat_topic():
    st.markdown("""
            <div style="
                background-color: #ffffff;
                border-radius: 5px;
                padding: 3px;
                margin-bottom: 3px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            ">
                <h3 style="
                    color: #333333;
                    font-size: 12px;
                    font-weight: bold;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                ">Chat Topic</h3>
            </div>
        """, unsafe_allow_html=True)


