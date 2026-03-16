# EEG Signal Classification Project

## Overview

This project focuses on analyzing Electroencephalography (EEG) signals and classifying brain activity using machine learning techniques. EEG signals capture the electrical activity of the brain and are widely used in neuroscience, healthcare, and brain–computer interface research.

The objective of this project is to process raw EEG recordings, extract meaningful features from the signals, and build a machine learning model capable of identifying patterns in brain activity.

## Project Objectives

* Process EEG recordings from raw data files
* Extract important frequency band features
* Train machine learning models for signal classification
* Evaluate model performance on EEG datasets
* Visualize EEG signals and extracted features

## Technologies Used

* Python
* NumPy
* SciPy
* Scikit-learn
* Matplotlib
* MNE (for EEG signal processing)

## Dataset

The EEG data used in this project is stored in **EDF (.edf) format**, which is commonly used for biomedical signal recordings. These files contain multi-channel brain signal recordings captured from EEG sensors.

## Methodology

The project follows the typical EEG signal processing pipeline:

1. Load EEG recordings from EDF files
2. Apply preprocessing techniques to remove noise and artifacts
3. Extract frequency band powers from the EEG signal:

   * Delta
   * Theta
   * Alpha
   * Beta
   * Gamma
4. Use the extracted features to train a machine learning classifier
5. Evaluate the model performance

## Project Structure

```
EEG/
│
├── data/                 # EEG data files
├── preprocessing/        # EEG signal preprocessing scripts
├── feature_extraction/   # Feature extraction from EEG signals
├── models/               # Machine learning models
├── results/              # Output results and visualizations
├── main.py               # Main program for running the pipeline
└── README.md
```

## Installation

Clone the repository:

```
git clone https://github.com/AishwaryaPrakash-code/-eeg-classification-project.git
```

Install the required dependencies:

```
pip install -r requirements.txt
```

## Running the Project

To run the EEG classification pipeline:

```
python main.py
```

The program will load EEG data, extract signal features, and perform classification.

## Applications

EEG signal classification has several real-world applications including:

* Brain–Computer Interface systems
* Neurological disorder detection
* Cognitive state monitoring
* Human–computer interaction research

## Future Improvements

Possible extensions of this project include:

* Using deep learning models for EEG classification
* Real-time EEG signal analysis
* Integration with wearable EEG devices

## Author

Aishwarya Prakash
Artificial Intelligence • Machine Learning • Robotics
