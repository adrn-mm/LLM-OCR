import streamlit as st
import requests
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from langchain.llms import AzureOpenAI
from langchain.agents import initialize_agent, AgentType

# load_dotenv()
# AZURE_COGS_KEY = os.getenv("AZURE_COGS_KEY")
# AZURE_COGS_ENDPOINT = os.getenv("AZURE_COGS_ENDPOINT")
# AZURE_COGS_REGION = os.getenv("AZURE_COGS_REGION")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

AZURE_COGS_KEY = st.secrets["AZURE_COGS_KEY"]
AZURE_COGS_ENDPOINT = st.secrets["AZURE_COGS_ENDPOINT"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_ENDPOINT = st.secrets["OPENAI_ENDPOINT"]

os.environ["AZURE_COGS_KEY"] = AZURE_COGS_KEY
os.environ["AZURE_COGS_ENDPOINT"] = AZURE_COGS_ENDPOINT 
os.environ["AZURE_COGS_REGION"] = "eastus"
 
def is_url_image(image_url):
    """ Memeriksa apakah URL adalah gambar. """
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    r = requests.head(image_url)
    if r.headers["content-type"] in image_formats:
        return True
    return False

def display_home():
    """ Menampilkan halaman Home. """
    st.header("LLM OCR")
    
    image_url = st.text_input("Masukkan link gambar (jpg, png, dll):", key="input_image_url")
    submit_button = st.button("Submit")

    if submit_button:
        if image_url.strip() == '':
            st.warning("Tolong masukkan link gambar.")
        elif is_url_image(image_url):
            st.session_state['image_url'] = image_url
            st.session_state['on_home_page'] = False
            st.experimental_rerun()  # Menjalankan ulang skrip dengan state terbaru
        else:
            st.error("URL tidak valid atau gambar tidak ditemukan.")

def display_chat_bot():
    """ Menampilkan halaman Chat Bot. """
    
    image_url = st.session_state['image_url']
    
    if st.sidebar.button("Refresh"):
        st.session_state['on_home_page'] = True
        st.session_state['image_url'] = ''
        st.experimental_rerun()
        
    st.sidebar.image(st.session_state['image_url'], use_column_width=True)
    
    agent = initialize_chatbot()

    user_question = st.text_input("Tanyakan sesuatu tentang gambar:")
    if st.button("Tanya"):
        response = agent.run(f'"{user_question}" "{image_url}"')
        st.write(response)

def initialize_chatbot():
    toolkit = AzureCognitiveServicesToolkit()

    llm = AzureOpenAI(azure_deployment="gpt-35-turbo",
                    openai_api_version="2023-05-15",
                    openai_api_key=OPENAI_API_KEY,
                    azure_endpoint=OPENAI_ENDPOINT)    
    
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    
    return agent
    

def main():
    """ Fungsi utama aplikasi Streamlit. """
    
    st.set_page_config(layout="wide",
                       page_title='MLPT - LLM OCR',
                       page_icon=':robot_face')
    
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.image('logo_mlpt.png')

    # Membuat state untuk menyimpan URL gambar dan status halaman
    if 'image_url' not in st.session_state:
        st.session_state['image_url'] = ''
    if 'on_home_page' not in st.session_state:
        st.session_state['on_home_page'] = True

    # Menampilkan halaman berdasarkan kondisi
    if st.session_state['on_home_page']:
        display_home()
    else:
        display_chat_bot()

if __name__ == "__main__":
    main()
