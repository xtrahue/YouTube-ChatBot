import streamlit as st
import json
import os
import whisper
import yt_dlp
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import Settings
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
import torch
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# Initialize session state

if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = ""
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}]
if 'is_generating_response' not in st.session_state:
    st.session_state.is_generating_response = False
if 'is_transcription_complete' not in st.session_state:
    st.session_state.is_transcription_complete = False
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""
if 'video_title' not in st.session_state:
    st.session_state.video_title = ""

model = whisper.load_model("small")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def download_audio(link):
    if os.path.exists('audio.mp3'):
        os.remove('audio.mp3')
    # st.session_state.transcription_result=""
    with yt_dlp.YoutubeDL({'extract_audio': True,
                           'format': 'bestaudio',
                           'outtmpl': 'audio.mp3'}) as video:
        info_dict = video.extract_info(link, download=True)
        st.session_state.video_title = info_dict['title']
        video_url = info_dict['webpage_url']
        video.download(link)
    return st.session_state.video_title, video_url


@st.cache_resource
def transcribe(_model, audio):
    if os.path.exists('text_files/transcription.txt'):
        os.remove('text_files/transcription.txt')

    result = _model.transcribe(audio)
    with open("text_files/transcription.txt", 'w') as f:
        f.write(result["text"])
    st.session_state.transcription_result = result["text"]
    return 1


def colud_config():
    cloud_config = {
        'secure_connect_bundle': 'secure-connect-yt-db.zip'
    }

    with open("mondal.indrajit7131@gmail.com-token.json") as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()

    load_dotenv()


def initialize_model():
    groq_api = os.environ["GROQ_API_KEY"]
    llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api)
    # st.write("model intialized")

    # Configuring Gradient Embeddings
    embed_model = GradientEmbedding(
        gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
        gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
        gradient_model_slug="bge-large",
    )
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 256

    # st.write("configured Gradient")

    return True


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}]


def main():
    if not st.session_state.is_initialized:
        colud_config()
        initialize_model()
        st.session_state.is_initialized = True

    with st.sidebar:
        st.title('🎦💬 YouTube Chatbot')
        url = st.text_input('Enter Youtube Video URL:')
        if not (url.startswith('https:')):
            st.warning('Please enter a valid URL!', icon='⚠️')
        else:
            st.success('Proceed to Submit!')
            if st.button("Submit") and url:
                st.session_state.video_title, st.session_state.video_url = download_audio(
                    url)
                transcribe(model, "audio.mp3")
                st.session_state.is_transcription_complete = True

    if not st.session_state.is_transcription_complete:
        st.warning('Please upload and process the YouTube video link.', icon='⚠️')
    else:
        st.success('Ready to answer your queries!', icon='👉')
        with st.sidebar:
            st.write("Video Title: "+st.session_state.video_title)
            st.video(st.session_state.video_url)

        # Rest of the code for chat functionality
        documents = SimpleDirectoryReader(
            "text_files").load_data()

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        if "query_engine" in st.session_state:
            query_engine = st.session_state["query_engine"]

        # Display initial message
        if len(st.session_state.messages) == 1:
            with st.chat_message(st.session_state.messages[0]["role"]):
                st.write(st.session_state.messages[0]["content"])

        # Display messages
        for message in st.session_state.messages[1:]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Handle user input and display messages
        if prompt := st.chat_input():
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if not st.session_state.is_generating_response:
                st.session_state.is_generating_response = True
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = query_engine.query(prompt).response
                        placeholder = st.empty()
                        placeholder.markdown(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
                st.session_state.is_generating_response = False

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


if __name__ == "__main__":
    main()
