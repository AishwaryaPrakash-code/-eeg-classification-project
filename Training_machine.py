import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

set_file ="C:/Users/aishw/PycharmProjects/EEG/normalized_eeg_chb.npy"
data = np.load(set_file)  # This should load your EEG data directly

# Step 2: Ensure data has correct shape: (n_epochs, n_channels, n_samples)
# The shape should be (n_epochs, n_channels, n_samples)
# For example, if data is 2D (e.g., (n_samples, n_channels)), you need to reshape it into 3D.
# Let's assume you have one epoch per sample:

# If the data shape is (n_samples, n_channels), reshape it into (n_samples, n_channels, 1)
if len(data.shape) == 2:
    data = data[..., np.newaxis]  # Add a new axis to make it (n_samples, n_channels, 1)

# Step 3: Apply bandpass filter (0.5 - 50 Hz)
n_channels = data.shape[1]  # Number of channels
sfreq = 250  # Sampling frequency (adjust based on your data)
info = mne.create_info(ch_names=[f"ch_{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg")

# Create MNE EpochsArray
epochs = mne.EpochsArray(data, info)

# Apply bandpass filter (0.5 - 50 Hz)
epochs.filter(l_freq=0.5, h_freq=50, fir_design='firwin')

# Step 4: Apply baseline correction
epochs.apply_function(lambda x: x - np.mean(x, axis=-1, keepdims=True))

# Step 5: Normalize data (Z-score normalization)
normalized_data = (epochs.get_data() - np.mean(epochs.get_data(), axis=1, keepdims=True)) / np.std(epochs.get_data(), axis=1, keepdims=True)

# Save the normalized data for future use
np.save("normalized_eeg_new.npy", normalized_data)
print("Normalization complete. File saved as 'normalized_eeg_new.npy'.")

# Step 6: Prepare data for training
# Reshape data for input to CNN (add channel dimension)
X = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], normalized_data.shape[2], 1)

# Example labels (binary classification, replace with actual labels)
labels = np.random.randint(0, 2, size=(normalized_data.shape[0],))

# Encode labels (one-hot encoding)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)

# Step 7: Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data prepared: Train={X_train.shape}, Test={X_test.shape}")

# Step 8: Define CNN model
model = models.Sequential()

# Add convolutional layers with smaller kernel sizes and padding to preserve dimensions
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1), padding='same'))
model.add(layers.MaxPooling2D((1, 2), padding='SAME'))  # Use smaller pool size

# Add more convolutional layers with smaller kernel sizes and padding
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((1, 2)))  # Use smaller pool size for second layer

# Use Global Average Pooling instead of Flatten
model.add(layers.GlobalAveragePooling2D())

# Add dense layer
model.add(layers.Dense(64, activation='relu'))

# Output layer for classification
model.add(layers.Dense(y.shape[1], activation='softmax'))  # 'softmax' for multi-class, 'sigmoid' for binary

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'binary_crossentropy' for binary classification
              metrics=['accuracy'])

# Step 9: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 10: Evaluate the model's performance on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Optional: Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()
model.save("EEG_chb_model.h5")
print("Model saved as 'EEG_chb_model.h5'")
