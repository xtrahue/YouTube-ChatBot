# YouTube Chatbot

The YouTube Chatbot is an AI-powered application that transcribes YouTube videos and uses the transcriptions to answer user queries in a chat format. It provides an interactive and engaging way to explore video content, making information more accessible.

## Features

* Transcribes YouTube videos using the Whisper model
* Processes transcriptions using Gradient Embeddings
* Generates responses to user queries based on the processed transcriptions using the Groq model
* Built with Streamlit for a user-friendly interface
* Allows users to submit a YouTube video URL and ask questions related to the video content
* Provides a clear chat history button to start a new conversation or switch to a different video

## Getting Started

To get started with the YouTube Chatbot, follow these steps:

1. Clone the repository: `git clone https://github.com/xtrahue/youtube-chatbot.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Set up your API keys and credentials for the Whisper, Gradient, and Groq models.
4. Run the application: `streamlit run app.py`

## Usage

1. Enter a YouTube video URL in the text input field and click the "Submit" button.
2. Wait for the video to be transcribed and processed.
3. Ask questions related to the video content in the chat input field.
4. The bot will generate responses based on the processed transcriptions.
5. Use the "Clear Chat History" button to start a new conversation or switch to a different video.

## Contributing

Contributions are welcome! If you would like to contribute to the YouTube Chatbot, please open a pull request with your proposed changes.

## License

The YouTube Chatbot is open-source and licensed under the MIT License.

## Acknowledgements

* Whisper: A general-purpose speech recognition model developed by OpenAI
* Gradient: A platform for building and deploying machine learning models
* Groq: A large language model for generating responses to user queries
* Streamlit: An open-source app framework for machine learning and data science applications.
