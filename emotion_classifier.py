import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import soundfile
from pydub import AudioSegment
import datetime
from sklearn.metrics import accuracy_score


class EmotionClassifier():
    
    def record_audio(self):
        CHUNK = 1024 
        FORMAT = pyaudio.paInt32 #paInt8
        CHANNELS = 1 
        RATE = 16000 #sample rate
        RECORD_SECONDS = 3

        basename = "live_audio"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        extent = ".wav"
        filename = "_".join([basename, suffix,extent])

        WAVE_OUTPUT_FILENAME = filename
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data) # 2 bytes(16 bits) per channel

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return filename;
    
    def classify_audio(self,filename):
        def extract_feature(file_name, mfcc, chroma, mel):
            with soundfile.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate=sound_file.samplerate
                if chroma:
                    stft=np.abs(librosa.stft(X))
                result=np.array([])
                if mfcc:
                    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack((result, mfccs))
                if chroma:
                    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                result=np.hstack((result, chroma))
                if mel:
                    mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))
            return result
        
        sound = AudioSegment.from_wav(filename)
        sound = sound.set_channels(1)
        sound.export(filename, format="wav")
        
        filename1 = filename
        file = filename1
        ans =[]
        new_feature  = extract_feature(file, mfcc=True, chroma=True, mel=True)
        ans.append(new_feature)
        ans = np.array(ans)

        loaded_model = pickle.load(open('./Emotion_Voice_Detection_Model.pkl', 'rb'))

        res = loaded_model.predict(ans)
        print('predicted',res)
        
        return res