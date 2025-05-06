# 🎬 YouTube Video AI Assistant

This project is an AI-powered web application that allows users to process YouTube videos by downloading, transcribing, summarizing, translating, and querying their content using natural language. The app provides an interactive interface for video analysis, translation, and chatbot interaction powered by OpenAI's GPT models and LangChain.

## 🚀 Project Goal

The goal of this system is to make YouTube video content more accessible by:
- Automatically transcribing spoken content into text
- Summarizing long transcripts into concise key points
- Enabling natural language querying of the video content
- Translating transcripts into multiple languages
- Providing audio playback of translated text

The app is designed for educational, accessibility, and content management use cases.

## 🏗️ Architecture

The system architecture includes:
1. **Video Downloader:** Uses `yt_dlp` to download YouTube videos.
2. **Audio Extractor:** Extracts audio from video using `ffmpeg`.
3. **Transcription Engine:** Uses `Whisper` for speech-to-text transcription.
4. **Summarization Module:** GPT-4 API generates concise summaries.
5. **Embedding & Vector Store:** `LangChain` with OpenAI Embeddings and `Chroma` store for semantic search.
6. **QA Agent:** LangChain agent handles user questions using transcript search.
7. **Translation & TTS:** GPT-4 translates transcript, OpenAI TTS generates translated audio.
8. **Frontend UI:** Built with `Gradio` with multi-tab interface for processing, chatting, and translation.

The system uses OpenAI API and LangChain to integrate large language models and retrieval-augmented generation.

## 🔬 Methodology

The system follows these steps:
1. User provides a YouTube video URL.
2. The video is downloaded and audio is extracted.
3. Audio is transcribed to generate a full transcript.
4. The transcript is summarized via GPT-4.
5. The transcript is split into chunks and embedded into a vector database.
6. Users can query the transcript via chatbot powered by LangChain agent.
7. The transcript can be translated into other languages and converted to speech.

The system was tested by processing multiple videos across different durations, languages, and topics to ensure stable downloading, transcription accuracy, and query performance.

## 📁 Repository Structure

project/
├── app.py # Main application code 


## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd project

2. **Create a virtual environment (optional but recommended):**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies:**
pip install -r requirements.txt

4. **Run the app:**
python app.py


## Requirements

Dependencies are listed in requirements.txt



## Usage Guide

✅ **Go to the Process Video tab:**
- Paste a YouTube URL.
- Click **"Process Video"** to download, transcribe, and summarize.
- View transcript and summary.

✅ **Go to the Chat tab:**
- Ask questions about the video content.

✅ **Go to the Translate tab:**
- Select target language.
- Click **"Translate"** to translate transcript.
- Click **"Play Audio"** to listen to translation.

## 💡 Notes
- Works best for videos in English or supported Whisper languages.
- Long videos may take more time for transcription and embedding.
- Translation supports languages listed in the dropdown.
- API key usage may incur costs depending on OpenAI plan.

## DEMO
https://drive.google.com/file/d/1_nMSd6OiETaWojVm58IjAKd1F6uUi8Iw/view?usp=sharing
Enjoy exploring video content with AI! 🌟
