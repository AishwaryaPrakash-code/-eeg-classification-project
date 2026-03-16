import numpy as np
import mne

# Load EEG data from .npy file
set_file = "C:/Users/aishw/PycharmProjects/EEG/merged_chb_data.npy"
data = np.load(set_file, allow_pickle=True)

# Print the shape of the data to debug
print(f"Shape of data: {data.shape}")

# Calculate number of samples per epoch
n_epochs = data.shape[0]  # 21 epochs
total_samples = data.shape[1]  # 1138000 samples

# Calculate the number of samples per epoch
samples_per_epoch = total_samples // n_epochs
print(f"Samples per epoch: {samples_per_epoch}")

# Try to detect the number of channels (n_channels) dynamically
# If we assume 23 channels, let's check
n_channels = 23

# Check if the total number of samples is divisible by the number of channels
if total_samples % n_channels != 0:
    print(f"Total samples {total_samples} are not divisible by the number of channels {n_channels}.")
    # Trim extra samples to make it divisible by n_channels
    new_total_samples = (total_samples // n_channels) * n_channels
    print(f"Trimmed total samples: {new_total_samples}")
    data = data[:, :new_total_samples]  # Trim the data

    # Recalculate the number of samples per channel
    n_samples_per_channel = new_total_samples // n_channels
else:
    n_samples_per_channel = total_samples // n_channels

# Print out the detected number of channels and samples
print(f"Detected number of channels: {n_channels}")
print(f"Samples per channel: {n_samples_per_channel}")

# Reshape the data into 3D: (n_epochs, n_channels, n_samples_per_channel)
data_reshaped = data.reshape(n_epochs, n_channels, n_samples_per_channel)

# Print the new shape to confirm it's correct
print(f"Reshaped data shape: {data_reshaped.shape}")

# Create MNE EpochsArray object if needed for filtering
info = mne.create_info(ch_names=[f"ch_{i}" for i in range(n_channels)], sfreq=250,
                       ch_types="eeg")  # Adjust `sfreq` as needed
epochs = mne.EpochsArray(data_reshaped, info)

# Apply bandpass filter (0.5 - 50 Hz)
epochs.filter(l_freq=0.5, h_freq=50, fir_design='firwin')

# Normalize each channel using Z-score normalization
normalized_data = (epochs.get_data() - np.mean(epochs.get_data(), axis=-1, keepdims=True)) / np.std(epochs.get_data(),
                                                                                                    axis=-1,
                                                                                                    keepdims=True)

# Save normalized data
np.save("normalized_eeg_chb.npy", normalized_data)
print("Normalization complete. File saved as 'normalized_eeg_chb.npy'.")
