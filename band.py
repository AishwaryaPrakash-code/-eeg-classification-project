import mne
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip
import os

# Load raw EEG data (EDF format)
file_path = "C:/Users/aishw/Downloads/Chb_seizures/chb01_43.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)
raw.pick_types(eeg=True)

# Create fixed-length epochs (2-second segments)
events = mne.make_fixed_length_events(raw, duration=2.0)
epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)

# Define EEG frequency bands
bands = {
    'Delta': (0.5, 4),    # Deep sleep
    'Theta': (4, 8),      # Drowsy, meditation
    'Alpha': (8, 12),     # Relaxed, calm
    'Beta': (12, 30),     # Active thinking, focus
    'Gamma': (30, 40)     # High-level cognition
}

# Calculate average band powers
band_powers = {}
for band, (fmin, fmax) in bands.items():
    filtered = epochs.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
    data = filtered.get_data(copy=True)
    band_powers[band] = data.mean(axis=(0, 1)).mean()

# Create explanation text with mental state interpretations
explanation = (
    "This graph displays the average EEG power across five frequency bands. "
    "The Delta band, between 0.5 and 4 Hertz, reflects deep sleep or unconscious states. "
    f"In this case, its power is approximately {band_powers['Delta']:.2f} microvolts squared. "
    "The Theta band, from 4 to 8 Hertz, is linked to drowsiness or meditation. "
    f"It shows a power of around {band_powers['Theta']:.2f}. "
    "Alpha waves, between 8 and 12 Hertz, represent a calm and relaxed state. "
    f"The power in this band is {band_powers['Alpha']:.2f}. "
    "Beta waves, from 12 to 30 Hertz, indicate alertness and mental activity. "
    f"It has a power of {band_powers['Beta']:.2f}. "
    "Finally, Gamma waves, from 30 to 40 Hertz, are associated with high-level cognitive functioning, with a power of "
    f"{band_powers['Gamma']:.2f}. These patterns help understand a person's mental and emotional state."
)

# Generate voice from text
tts = gTTS(text=explanation)
audio_path = "band_power_explanation.mp3"
tts.save(audio_path)

# Save graph as an image
plt.figure(figsize=(8, 6))
plt.bar(band_powers.keys(), band_powers.values(), color='skyblue')
plt.title("EEG Band Powers and Their Mental States")
plt.xlabel("EEG Frequency Band")
plt.ylabel("Average Power (µV²)")
plt.tight_layout()
image_path = "band_powers_graph.png"
plt.savefig(image_path)
plt.close()

# Create video (image + audio)
audio_clip = AudioFileClip(audio_path)
image_clip = ImageClip(image_path).set_duration(audio_clip.duration).set_audio(audio_clip)
image_clip = image_clip.set_fps(1)  # For static image video
output_video = "band_power_with_voice1.mp4"
image_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

print("🎬 Final video saved as:", output_video)
