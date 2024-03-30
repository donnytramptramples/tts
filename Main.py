import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

# Specify model path (replace with your actual .bin file location)
model_path = "your_model_path.bin"

# Load the model from the .bin file
model = Wav2Vec2ForCTC.from_pretrained(model_path)

# Load the tokenizer associated with the model (assuming it's included in the .bin file)
tokenizer = Wav2Vec2Processor.from_pretrained(model_path)  # Might require adjustment

def synthesize_speech(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    # Generate audio waveform using librosa
    sample_rate = tokenizer.model_kwargs.sampling_rate  # Get sample rate from tokenizer
    audio_embedding = model(**inputs).encoder_outputs[0]  # Extract audio embedding
    mel_spectrogram = librosa.feature.melspectrogram(audio_embedding.cpu().numpy().squeeze())
    griffin_lim = librosa.effects.griffin_lim(mel_spectrogram, sr=sample_rate)

    return griffin_lim

# Get user input
text = input("Type the text you want to convert to speech: ")

# Synthesize speech and save audio
waveform = synthesize_speech(text)
librosa.output.write_wav("output.wav", waveform, tokenizer.model_kwargs.sampling_rate)

print("Audio generated and saved to output.wav")
