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
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #a1c4fd 100%);
        color: white; /* Color global del texto (puedes modificar según lo que necesites) */
    }

    /* Buttons styling */
    div.stButton > button {
        color: #ffffff;
        background-color: #1e3c72;
        font-size: 16px;
        border-radius: 5px;
        border: none;
    }
    div.stButton > button:hover {
        color: #ffffff;
        background-color: #2a5298;
    }
    
    /* Navigation info column styling (por ejemplo, si se muestra en el sidebar) */
    [data-testid="stSidebar"] {
        background: #25306c; /* Fondo azul oscuro */
    }
    [data-testid="stSidebar"] * {
        color: white !important; /* Forzamos que todo el texto sea blanco */
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
