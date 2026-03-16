import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import mne

# Replace this with the actual path to your EEG .set file
eeg_file = r"C:/Users/aishw/OneDrive/Desktop/E/sub01.set"

# Load the data as epochs (without preload argument)
epochs = mne.io.read_epochs_eeglab(eeg_file)

# Access data and times
data = epochs.get_data(copy=True)  # Shape will be (n_epochs, n_channels, n_times)
times = epochs.times  # Times associated with the EEG data

# Print some basic info
print(f"Data shape: {data.shape}")
print(f"First few time points: {times[:10]}")

# Video settings
fps = 30  # Frames per second
duration = 10  # Duration in seconds
frame_count = fps * duration
output_video = "eeg_video.mp4"
width, height = 800, 600

# Create a temporary directory for frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

# Normalize data for visualization
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Create and save frames for the first 5 epochs (for example)
for epoch_idx in range(5):  # Modify this loop to visualize more epochs if desired
    plt.figure(figsize=(10, 5))
    plt.plot(times, data[epoch_idx].T)  # Plot EEG over time for each epoch
    plt.xlabel("Time (s)")
    plt.ylabel("EEG Signal")
    plt.title(f"EEG Data for Epoch {epoch_idx + 1}")
    plt.grid()

    frame_path = f"{frames_dir}/frame_{epoch_idx:04d}.png"
    plt.savefig(frame_path)
    plt.close()

# Create video from frames
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for epoch_idx in range(5):  # Modify this loop to visualize more epochs if desired
    frame = cv2.imread(f"{frames_dir}/frame_{epoch_idx:04d}.png")
    frame = cv2.resize(frame, (width, height))
    video.write(frame)

video.release()
cv2.destroyAllWindows()

# Cleanup: Remove the temporary frames directory
for file in os.listdir(frames_dir):
    os.remove(os.path.join(frames_dir, file))
os.rmdir(frames_dir)

print(f"Video saved as {output_video}")
