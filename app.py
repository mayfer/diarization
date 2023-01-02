import whisper
import datetime
import subprocess
import gradio as gr
from pathlib import Path
import pandas as pd
import re
import time
import os 
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pytube import YouTube
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

import psutil
num_cores = psutil.cpu_count()
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"

whisper_models = ["base", "small", "medium", "large", "base.en"]
source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

source_language_list = [key[0] for key in source_languages.items()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE IS: ")
print(device)

videos_out_path = Path("./videos_out")
videos_out_path.mkdir(parents=True, exist_ok=True)


def time(secs):
    return datetime.timedelta(seconds=round(secs))

def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print("Success download video")
    print(abs_video_path)
    return abs_video_path


def speech_to_text(video_file_path, selected_source_lang, whisper_model, num_speakers):
    """
    # Youtube with translated subtitles using OpenAI Whisper
    This space allows you to:
    1. Download youtube video with a given url
    2. Watch it in the first video component
    3. Run automatic speech recognition and diarization (speaker identification)
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """
    
    model = whisper.load_model(whisper_model)
    if(video_file_path == None):
        raise ValueError("Error no video input")
    print(video_file_path)

    try:
        # Read and convert youtube video
        _,file_ending = os.path.splitext(f'{video_file_path}')
        print(f'file enging is {file_ending}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        print("starting conversion to wav")
        os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
        
        # Get duration
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(audio_file, **transcribe_options)
        segments = result["segments"]
        print("starting whisper done with whisper")
    except Exception as e:
        raise RuntimeError("Error converting video to audio")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(time(segments[i - 1]["end"])))
        objects['Text'].append(text)
        
        return pd.DataFrame(objects)
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
video_in = gr.Video(label="Video file", mirror_webcam=False)
youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
video_out = gr.Video(label="Video Out", mirror_webcam=False)


df_init = pd.DataFrame(columns=['Start','End', 'Speaker', 'Text'])

selected_source_lang = gr.Dropdown(choices=source_language_list, type="value", value="en", label="Spoken language in video", interactive=True)
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="base", label="Selected Whisper model", interactive=True)
number_speakers = gr.Number(precision=0, value=2, label="Selected number of speakers", interactive=True)

transcription_df = gr.DataFrame(value=df_init,label="Transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')

demo = gr.Blocks(css='''
#cut_btn, #reset_btn { align-self:stretch; }
#\\31 3 { max-width: 540px; }
.output-markdown {max-width: 65ch !important;}
''')
demo.encrypt = False


with demo:
    transcription_var = gr.Variable()
    
    with gr.Row():
        with gr.Column():
            gr.Markdown('''
            ### This space allows you to: 
            ##### 1. Download youtube video with a given URL
            ##### 2. Watch it in the first video component
            ##### 3. Run automatic speech recognition and diarization (speaker identification)
            ''')
            memory = psutil.virtual_memory()
            system_info = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")
        
        with gr.Column():
            gr.Markdown('''
            ### Insert Youtube URL below. Some test youtube links below:
            ''')
            examples = gr.Examples(examples=
                [ "https://www.youtube.com/watch?v=j7BfEzAFuYc&t=32s", 
                  "https://www.youtube.com/watch?v=-UX0X45sYe4", 
                  "https://www.youtube.com/watch?v=7minSgqi-Gw"],
               label="Examples", inputs=[youtube_url_in])
           

            
    with gr.Row():
        with gr.Column():
            youtube_url_in.render()
            download_youtube_btn = gr.Button("Download Youtube video")
            download_youtube_btn.click(get_youtube, [youtube_url_in], [
                video_in])
            print(video_in)
            

    with gr.Row():
        with gr.Column():
            video_in.render()
            with gr.Column():
                gr.Markdown('''
                ##### Here you can start the transcription process.
                ##### Please select the source language for transcription.
                ##### You should select a number of speakers for getting better results.
                ''')
            selected_source_lang.render()
            selected_whisper_model.render()
            number_speakers.render()
            transcribe_btn = gr.Button("Transcribe audio and diarization")
            transcribe_btn.click(speech_to_text, [video_in, selected_source_lang, selected_whisper_model, number_speakers], transcription_df)

            
    with gr.Row():
        gr.Markdown('''
        ##### Here you will get transcription  output
        ##### ''')

    with gr.Row():
        with gr.Column():
            transcription_df.render()

demo.launch(debug=True)