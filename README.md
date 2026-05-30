# Emotion Based Music Recommendation System

This repository contains a Streamlit application that scans a user's facial expression through a webcam, predicts dominant emotions with a pretrained CNN model, and generates a playlist-style recommendation set from the bundled `muse_v3.csv` dataset.

## Stack

- Streamlit
- OpenCV
- TensorFlow / Keras
- pandas
- NumPy

## Features

- Live webcam-based emotion scanning
- Emotion ranking across multiple captured frames
- Mood-bucket normalization for playlist generation
- Recommendation count controls from the sidebar
- Recommendation export as CSV
- Emotion breakdown summary after each scan

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. The app opens a webcam session through OpenCV.
2. A pretrained emotion model predicts facial emotion per detected face frame.
3. The captured emotions are ranked by frequency.
4. The top mood buckets are mapped to segments of the music dataset.
5. A recommendation list is sampled and displayed in the Streamlit interface.

## Modernization Pass

This version keeps the original idea but updates the experience:

1. removed brittle top-level webcam handling
2. fixed modern pandas compatibility
3. added caching for large resources
4. added sidebar controls for scan depth and playlist size
5. added an emotion-count summary and recommendation export
6. improved the visual presentation to feel more current

## Notes

- This project still depends on a working webcam.
- The pretrained `model.h5` file is expected to stay in the repository root.
- Recommendation quality depends on both the emotion model and the quality of the source song dataset.
- The app performs real recommendation generation, but the playlist logic is still heuristic rather than personalized.
