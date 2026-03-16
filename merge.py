import mne
import glob
import os

# Path where EDF files are stored
edf_folder = "C:/Users/aishw/Downloads/Chb_seizures"

edf_files = sorted(glob.glob(os.path.join(edf_folder, "*.edf")))

all_data = []

for file_path in edf_files:
    print(f"Processing {os.path.basename(file_path)}...")

    # Load EEG data
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.rename_channels({name: f"{name}_{i}" for i, name in enumerate(raw.ch_names) if raw.ch_names.count(name) > 1})
    all_data.append(raw)

    print(f"Data shape for {os.path.basename(file_path)}: {raw.get_data().shape}")

# Merge all raw data if needed
if all_data:
    merged_raw = mne.concatenate_raws(all_data)
    print("All files merged successfully!")

output_edf_path = "C:/Users/aishw/PycharmProjects/EEG/merged_chb_data.edf"
mne.export.export_raw(output_edf_path, merged_raw, fmt='edf', overwrite=True)
print(f"Merged data saved as EDF: {output_edf_path}")
