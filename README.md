# SONIQ — Your Music, Decoded 

### Machine Learning Based Music Analysis and Beat-Responsive Visual Intelligence

SONIQ is a machine-learning-based music analysis web application that analyses uploaded audio, estimates its emotional characteristics, identifies musical patterns, and generates music improvement suggestions. It combines audio signal processing, machine learning, and interactive visualisation in a single interface.

##  Features

### 1. Audio Upload and Analysis
- Upload audio files in supported formats such as MP3 and WAV.
- Analyse musical characteristics using audio signal processing.
- Extract features including:
  - Tempo (BPM)
  - Beat information
  - Duration
  - Root Mean Square (RMS) energy
  - Spectral centroid
  - Zero-crossing rate

### 2. ML-Based Emotion Prediction
- Predicts two continuous emotional dimensions:
  - **Valence:** Estimated musical positivity.
  - **Arousal:** Estimated musical energy or intensity.
- Uses a Random Forest regression model through `MultiOutputRegressor`.
- Displays model predictions alongside audio-analysis results.

### 3. Beat-Responsive Visualisation
- Provides an interactive audio visualisation experience.
- Displays beat markers and audio-intensity events.
- Includes a timeline for exploring different parts of a track.
- Supports seeking through the audio using timeline events.

### 4. Music Improvement Suggestions
- Provides rule-based suggestions based on extracted audio characteristics.
- Highlights areas that may benefit from further musical development.

### 5. Music Development Pathway
- Presents structured guidance to help users explore potential improvements in their music.
- Connects audio-analysis results with practical development suggestions.

### 6. Music Comparison
- Compare two audio files using their extracted audio characteristics and predicted emotion scores.
- Explore differences between two tracks or different versions of the same song.
- View a comparison of selected musical metrics.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web framework | Flask |
| Machine learning | Scikit-learn |
| Audio analysis | Librosa |
| Audio feature extraction | openSMILE |
| Numerical processing | NumPy |
| Data processing | Pandas |
| Frontend | HTML, CSS, JavaScript |
| Audio visualisation | Web Audio API |

##  Machine Learning Approach

SONIQ uses a multi-output regression approach to estimate valence and arousal from audio-derived features.

The model was trained using the DEAM (Database for Emotion Analysis using Music) dataset.

### Model Configuration
- **Estimator:** Random Forest Regressor
- **Wrapper:** MultiOutputRegressor
- **Training samples:** 1,441
- **Test samples:** 361
- **Feature dimensions:** 520

### Recorded Evaluation Results

| Metric | Valence | Arousal |
|---|---:|---:|
| Mean Absolute Error (MAE) | 0.697 | 0.615 |
| R² Score | 0.418 | 0.583 |

These results are from the recorded model evaluation. Predictions are estimates rather than definitive interpretations of musical emotion. The compatibility between the training and inference feature pipelines requires further validation.

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.10 or a compatible Python version
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/SrivalliOduri/soniq-music-analysis.git
cd soniq-music-analysis
```

### 2. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Some audio-processing dependencies may require additional system libraries, depending on your operating system.

### 4. Run the Application

```bash
python app.py
```

Open the local URL printed in the terminal, typically:

```text
http://127.0.0.1:5000
```

**Note:** The application also requires its trained model files and any other project assets expected by the source code. Follow the repository's setup instructions if additional configuration is required.

##  Project Structure

The exact structure depends on the files included in the repository. A typical layout is:

```text
soniq-music-analysis/
├── app.py
├── requirements.txt
├── README.md
├── templates/
├── static/
├── models/
└── .gitignore
```

The `templates/`, `static/`, and `models/` directories are illustrative; retain the actual structure of your project rather than creating empty folders just to match this example.

##  Testing and Limitations

- Audio analysis depends on the quality and characteristics of the uploaded audio.
- Valence and arousal predictions may not reflect every listener's emotional interpretation.
- Training and inference feature compatibility requires further validation.
- Beat and intensity detection are estimates and may not perfectly match human perception.
- Improvement suggestions are rule-based guidance, not guaranteed outcomes.
- Comparison results describe differences in measured features and do not establish that two files are versions of the same song.

##  Future Enhancements

- Validate and improve the consistency of training and inference features.
- Evaluate the model on additional unseen audio samples.
- Improve emotion-prediction performance through feature engineering and model comparison.
- Enhance beat and musical-section detection.
- Expand personalised music-development recommendations.
- Improve deployment reliability and automated testing.

##  Project Objective

The objective of SONIQ is to combine machine learning, audio signal processing, and interactive visualisation to make music characteristics easier to understand and explore.

##  Project Information

**Project:** SONIQ — Your Music, Decoded  
**Domain:** Machine Learning, Audio Signal Processing and Interactive Visualisation  
**Application:** Music Analysis and Beat-Responsive Visual Intelligence

##  Disclaimer

SONIQ is an academic prototype developed for educational and experimental purposes. Its outputs are intended to support music exploration and should not be treated as definitive musical, emotional, or professional production assessments.

##  License

Add a license file if you intend to distribute the project under a specific open-source license. No license is implied unless one is explicitly included in the repository.
