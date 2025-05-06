from multiprocessing.connection import Client
import os
import tempfile
import gradio as gr
import whisper
import yt_dlp
import ffmpeg
from PIL import Image
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from fuzzywuzzy import process
from langsmith import Client
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_api_key = "your key"
client = OpenAI(api_key=openai_api_key)
os.environ["LANGCHAIN_API_KEY"] = "your key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "VideoQA-Agent"
langsmith_client = Client()

# Globals
whisper_model = whisper.load_model("small")
vector_store = None
agent = None
full_transcript = ""
chat_history = []
chroma_persist_dir = "./chroma_db"

def initialize_chroma(transcript_text):
    global vector_store
    
    # Remove old database if exists
    if os.path.exists(chroma_persist_dir):
        try:
            shutil.rmtree(chroma_persist_dir)
        except Exception as e:
            logger.error(f"Error removing old Chroma DB: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(transcript_text)
    
    docs = [Document(page_content=text) for text in texts]
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=chroma_persist_dir
    )
    vector_store.persist()

def download_youtube_video(url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "%(title)s.%(ext)s")
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': output_path,
            'quiet': True,
            'merge_output_format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info_dict)
            return filename, info_dict.get("thumbnail")
    except Exception as e:
        logger.error(f"Video download error: {str(e)}")
        raise gr.Error(f"Failed to download video: {str(e)}")

def extract_audio(video_path):
    try:
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        ffmpeg.input(video_path).output(audio_path, format='mp3', acodec='libmp3lame', ar='44100').run(overwrite_output=True, quiet=True)
        return audio_path
    except Exception as e:
        logger.error(f"Audio extraction error: {str(e)}")
        raise gr.Error(f"Failed to extract audio: {str(e)}")

def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        global full_transcript
        full_transcript = " ".join([seg['text'] for seg in result['segments']])
        return full_transcript
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise gr.Error(f"Failed to transcribe audio: {str(e)}")

def summarize_with_gpt(text):
    try:
        prompt = f"""
You are a professional summarizer. Summarize the following transcript into clear and concise English key points:

Transcript:
{text}
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"Failed to generate summary: {str(e)}"

def build_agent():
    tools = [Tool(name="VideoQA", func=search_video_tool, description="Answer ONLY from video transcript")]
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True)

def search_video_tool(query):
    global vector_store, full_transcript
    if vector_store is None:
        return full_transcript
    
    docs = vector_store.similarity_search(query, k=3)
    if docs:
        return "\n\n".join([doc.page_content for doc in docs])
    return full_transcript

def answer_with_agent(q):
    global agent, chat_history
    if agent is None:
        agent = build_agent()
    context = search_video_tool(q)
    if not context or len(context) < 20:
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": "This topic isn't covered in the video."})
        return chat_history
    prompt = f"Answer based STRICTLY on this video content:\n{context}\n\nQuestion: {q}"
    answer = agent.run(prompt)
    chat_history.append({"role": "user", "content": q})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

def check_transcript_length():
    global full_transcript
    if len(full_transcript.split()) > 10000:
        return "Warning: Very long transcript may cause translation issues"
    return None

def complete_translation(lang):
    try:
        max_chunk_size = 3000
        if len(full_transcript) > max_chunk_size:
            chunks = [full_transcript[i:i+max_chunk_size] for i in range(0, len(full_transcript), max_chunk_size)]
            translated_chunks = []
            
            for chunk in chunks:
                prompt = f"""Translate the following text to {lang} accurately and professionally.
Maintain the original meaning, tone, and technical terms where appropriate.
Do not summarize or omit any details.

Text to translate:
{chunk}"""
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                translated_chunks.append(response.choices[0].message.content.strip())
            
            return "\n\n".join(translated_chunks)
        else:
            prompt = f"""Translate the following text to {lang} accurately and professionally.
Maintain the original meaning, tone, and technical terms where appropriate.
Do not summarize or omit any details.

Text to translate:
{full_transcript}"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"Translation failed: {str(e)}"

def generate_openai_tts(text):
    try:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    except Exception as e:
        logger.error(f"TTS generation error: {str(e)}")
        return None

def process_all(url):
    global full_transcript, vector_store, agent, chat_history
    
    # Reset global variables
    full_transcript = ""
    vector_store = None
    agent = None
    chat_history = []
    
    try:
        video_path, thumbnail_url = download_youtube_video(url)
        audio_path = extract_audio(video_path)
        full_transcript = transcribe_audio(audio_path)
        summary = summarize_with_gpt(full_transcript)
        initialize_chroma(full_transcript)
        return thumbnail_url, summary, full_transcript
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return None, f"Error processing video: {str(e)}", ""

with gr.Blocks(theme=gr.themes.Base(primary_hue="orange")) as demo:
    gr.Markdown("# 🎬 YouTube Video AI Assistant")

    with gr.Tab("📜 Process Video"):
        thumbnail_img = gr.Image(label="Video Thumbnail", interactive=False)
        with gr.Row():
            url_link = gr.Textbox(label="YouTube URL")
        submit_btn = gr.Button("🚀 Process Video")
        summary_box = gr.Textbox(label="Summary", lines=5)
        transcript_box = gr.Textbox(label="Transcript", lines=10)

    with gr.Tab("🤖 Chat"):
        chatbot = gr.Chatbot(height=400, type="messages")
        question = gr.Textbox(label="Ask about the video")
        submit_btn2 = gr.Button("Send")
        clear_btn = gr.Button("🗑️")
        submit_btn2.click(fn=answer_with_agent, inputs=[question], outputs=[chatbot])
        clear_btn.click(lambda: [], None, [chatbot], queue=False)

    with gr.Tab("🌍 Translate"):
        with gr.Row():
            target_language = gr.Dropdown(
                choices=["Arabic", "English", "Spanish", "French", "German", "Italian", "Russian", "Chinese"],
                label="Select Language",
                  value="Arabic"
            )
            translate_btn = gr.Button("🌐 Translate")
            play_audio_btn = gr.Button("🔊 Play Audio")
        
        warning_box = gr.Textbox(label="Warning", visible=False)
        translation_output = gr.Textbox(label="Translated Text", lines=10)
        translation_audio = gr.Audio(label="Audio", type="filepath", visible=False)

        def handle_translation(lang):
            warning = check_transcript_length()
            if warning:
                return warning, None, warning
            
            try:
                translated = complete_translation(lang)
                if translated.startswith("Translation failed:"):
                    return translated, None, translated
                
                audio = generate_openai_tts(translated)
                if audio:
                    return translated, audio, None
                else:
                    return translated, None, "Failed to generate audio"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, None, error_msg

        translate_btn.click(
            handle_translation,
            [target_language],
            [translation_output, translation_audio, warning_box]
        )
        
        def toggle_audio_visibility(translation):
            return gr.Audio(visible=translation and not translation.startswith("Error:") and not translation.startswith("Translation failed:"))
        
        translation_output.change(
            toggle_audio_visibility,
            [translation_output],
            [translation_audio]
        )

        play_audio_btn.click(
            generate_openai_tts,
            [translation_output],
            [translation_audio]
        )

    submit_btn.click(
        fn=process_all,
        inputs=[url_link],
        outputs=[thumbnail_img, summary_box, transcript_box]
    )

demo.launch()




