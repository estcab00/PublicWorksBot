import streamlit as st

st.set_page_config(page_title="Chatbot 💬", layout="centered")

# --- PAGE SETUP ---
about_page = st.Page(
    page="views/chatbot.py",
    title="Chatbot",
    icon="💬",
    default=True,
)

# --- INJECT GLOBAL CSS FOR GRADIENT BACKGROUND ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #a1c4fd 100%);
        color: black; /* Ensure text is readable */
    }

    /* Change the text color of buttons */
    div.stButton > button {
        color: #ffffff; /* White text */
        background-color: #1e3c72; /* Match the darker gradient color */
        font-size: 16px; /* Adjust text size */
        border-radius: 5px; /* Add rounded corners */
        border: none; /* Clean button borders */
    }

    /* Change the button hover effect */
    div.stButton > button:hover {
        color: #ffffff; /* White text on hover */
        background-color: #2a5298; /* Slightly lighter shade for hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# --- NAVIGATION ---
pg = st.navigation(
    {
        "Info" : [about_page],
    }
)

# --- RUN NAVIGATION ---
pg.run()

# --- SIDEBAR ---
#st.sidebar.text("Made by @estcab00")
