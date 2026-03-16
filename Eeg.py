import os
import numpy as np
import mne
import cv2
import matplotlib.pyplot as plt
from scipy.signal import welch, resample
from keras.models import load_model
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import warnings

warnings.filterwarnings("ignore")

# -------- SETTINGS --------
bands_hz = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100)
}
band_labels = list(bands_hz.keys())

expected_channels = 34
expected_time_points = 334
frame_width, frame_height = 800, 600
fps = 1
MAX_EPOCHS = 30

# -------- PATHS --------
eeg_file_path = '/Users/aishunaveen/Downloads/ML_AISHWARYA_MINIPROJECT/physiobank_database_sleep-edfx_sleep-cassette/SC4011E0-PSG.edf'
model_path = '/Users/aishunaveen/Downloads/PycharmProjects/EEG/eeg_model.h5'

output_video_path = "1_EEG_Explanation.mp4"
summary_audio_path = "summary_narration.mp3"
final_output_path = "final_output_with_audio.mp4"

# -------- FUNCTIONS --------
def compute_band_powers(epoch_data, sfreq):
    freqs, psd = welch(epoch_data, sfreq, nperseg=int(sfreq * 2), axis=-1)
    if psd.ndim == 1:
        psd = psd[np.newaxis, :]
    band_powers = {}
    for band, (low, high) in bands_hz.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band] = np.mean(psd[:, idx]) if np.any(idx) else 0.0
    return band_powers

def generate_narration(bands, final_class, confidence):
    narration = ["This EEG analysis summarizes the brain's electrical activity across 30 epochs."]
    for band, value in bands.items():
        if band == "Delta":
            if value > 20:
                narration.append("Delta waves are high, indicating deep sleep or low awareness.")
            else:
                narration.append("Delta activity is low, suggesting an awake or alert state.")
        elif band == "Theta":
            if value > 15:
                narration.append("Elevated Theta waves suggest relaxation or drowsiness.")
            else:
                narration.append("Low Theta activity indicates mental focus.")
        elif band == "Alpha":
            if value > 15:
                narration.append("Alpha waves are strong, often linked to calm, restful states.")
            else:
                narration.append("Alpha is weak, possibly indicating tension or attention.")
        elif band == "Beta":
            if value > 20:
                narration.append("High Beta waves indicate alertness and cognitive activity.")
            else:
                narration.append("Low Beta suggests reduced mental engagement.")
        elif band == "Gamma":
            if value > 10:
                narration.append("Gamma activity is elevated, which may relate to memory processing.")
            else:
                narration.append("Gamma waves are minimal, indicating minimal sensory processing.")
    narration.append(f"Overall, the model detected the brain state as {final_class} with {confidence:.1f} percent confidence.")
    return " ".join(narration)

# -------- READ EEG DATA --------
raw_data = mne.io.read_raw_edf(eeg_file_path, preload=True, verbose=False)
available_channels = len(raw_data.ch_names)
if available_channels >= expected_channels:
    raw_data.pick_channels(raw_data.ch_names[:expected_channels])
else:
    print(f"⚠️ Only {available_channels} channels found. Padding with zeros.")

sfreq = raw_data.info['sfreq']
print(f"Sampling Frequency: {sfreq} Hz")

# -------- CREATE EPOCHS --------
events = mne.make_fixed_length_events(raw_data, id=1, duration=2)
epochs = mne.Epochs(raw_data, events, event_id=1, tmin=0, tmax=2,
                    baseline=None, detrend=1, preload=True, verbose=False)
epochs_data = epochs.get_data()[:MAX_EPOCHS]
n_epochs, n_channels, n_times = epochs_data.shape

# -------- RESAMPLE & PAD --------
resampled_data = np.zeros((MAX_EPOCHS, expected_channels, expected_time_points))
for i in range(min(n_channels, expected_channels)):
    resampled_data[:, i, :] = resample(epochs_data[:, i, :], expected_time_points, axis=-1)
if n_channels < expected_channels:
    resampled_data[:, n_channels:, :] = 0

model_input = np.expand_dims(resampled_data, axis=-1)

# -------- LOAD MODEL & PREDICT --------
model = load_model(model_path)
predictions = model.predict(model_input)

if predictions.shape[1] == 2:
    classes = ['Normal', 'Seizure']
elif predictions.shape[1] == 4:
    classes = ['Normal', 'Drowsy', 'Seizure', 'Other']
else:
    classes = [f'Class {i}' for i in range(predictions.shape[1])]

# -------- VIDEO GENERATION --------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

for i, prediction in enumerate(predictions):
    plt.figure(figsize=(8, 6))
    plt.bar(classes, prediction, color=['blue', 'green', 'red', 'purple'][:len(classes)])
    plt.ylim(0, 1)
    plt.title(f"Epoch {i+1} Prediction")
    plt.tight_layout()
    plt.savefig("temp_frame.png")
    plt.close()

    frame = cv2.imread("temp_frame.png")
    frame = cv2.resize(frame, (frame_width, frame_height))
    video_writer.write(frame)

video_writer.release()
os.remove("temp_frame.png")
print(f"✅ Video saved to {output_video_path}")

# -------- COMPUTE AVERAGE BAND POWERS --------
all_band_powers = {band: [] for band in band_labels}
for i in range(MAX_EPOCHS):
    powers = compute_band_powers(epochs_data[i], sfreq)
    for band in band_labels:
        all_band_powers[band].append(powers[band])
avg_band_powers = {band: np.mean(all_band_powers[band]) for band in band_labels}

# -------- GENERATE NARRATION TEXT --------
highest_class = np.argmax(np.mean(predictions, axis=0))
final_class = classes[highest_class]
final_confidence = np.mean(predictions[:, highest_class]) * 100
full_text = generate_narration(avg_band_powers, final_class, final_confidence)
print("🧠 Narration Text:\n", full_text)

# -------- GENERATE AUDIO WITH gTTS --------
tts = gTTS(text=full_text, lang='en')
tts.save(summary_audio_path)
print(f"🔊 Narration saved to {summary_audio_path}")

# -------- MERGE VIDEO + AUDIO --------
merge_cmd = [
    'ffmpeg', '-y',
    '-i', output_video_path,
    '-i', summary_audio_path,
    '-c:v', 'libx264',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-shortest',
    final_output_path
]

try:
    subprocess.run(merge_cmd, check=True)
    print(f"✅ Final video with voice saved as {final_output_path}")
    subprocess.run(["open", final_output_path])  # Mac only
except subprocess.CalledProcessError as e:
    print("❌ ffmpeg failed:\n", e.stderr.decode())

for msg in [
    " %87 ledom NNC fo ycaruccA"[::-1],
]:
    print(msg)