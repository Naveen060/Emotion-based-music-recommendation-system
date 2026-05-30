# Emotion Based Music Recommendation System

This repository contains a Streamlit application that scans a user's facial expression through a webcam, predicts the dominant emotion with a pretrained CNN model, and recommends songs from the included `muse_v3.csv` dataset.

## Current Project Structure

```text
Emotion-based-music-recommendation-system/
|-- app.py
|-- requirements.txt
|-- README.md
|-- muse_v3.csv
|-- model.h5
`-- haarcascade_frontalface_default.xml
```

## Features

- Webcam-based facial emotion scanning with OpenCV.
- Emotion classification using a pretrained TensorFlow / Keras model.
- Song recommendation using the included music dataset.
- Streamlit UI for running the scan and viewing recommendations.

## Changes Made In This Recovery Pass

The original repository contained a usable prototype, but it had several problems that would break or degrade the app on a modern Python stack. The following changes were applied:

1. Rebuilt the app around explicit helper functions.
   The original file relied heavily on top-level execution, which made it harder to validate, reuse, and recover.

2. Removed webcam initialization at import time.
   The old code opened `cv2.VideoCapture(0)` globally. The app now opens the webcam only when the user clicks `Scan Emotion`.

3. Fixed modern pandas compatibility.
   The original recommendation logic used `DataFrame.append`, which is removed in current pandas versions. The app now uses `pd.concat`.

4. Added file-path safety with `pathlib`.
   All key project files are now loaded relative to the repository path instead of the current working directory.

5. Cached large resources safely.
   The dataset, cascade classifier, and TensorFlow model are now loaded through Streamlit caching helpers so repeated reruns do not reload everything from disk each time.

6. Normalized emotion mapping for recommendations.
   The original code mixed labels such as `fear`, `happy`, `Neutral`, and `Fearful`, which caused several predictions to fall through to the wrong recommendation bucket. The app now maps prediction labels consistently before sampling songs.

7. Added runtime validation and clearer error handling.
   The recovered app now checks for:
   - missing dataset file
   - missing model weights
   - invalid dataset columns
   - missing cascade classifier
   - unavailable webcam
   - unreadable webcam frames

8. Preserved recommendations across Streamlit reruns.
   The app now uses `st.session_state` so the detected emotion summary and generated recommendations stay visible after the scan completes.

9. Cleaned the dependency list.
   The old `requirements.txt` included `collection`, which is not a valid installable package. The dependencies were reduced to the packages the app actually needs.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit application.

```powershell
streamlit run app.py
```

## Usage

1. Start the app with Streamlit.
2. Click `Scan Emotion`.
3. Allow the webcam to capture your face for several frames.
4. Press `x` in the OpenCV window if you want to stop the scan early.
5. Review the generated song recommendations in the Streamlit page.

## Notes

- This project requires a working webcam.
- The pretrained model file `model.h5` is expected to be present in the repository root.
- Recommendation quality depends on both model accuracy and the quality of the source song dataset.
- This recovery pass focused on code stability and maintainability, not on retraining the emotion model.

## Verification Performed

- The repository was cloned locally and inspected.
- `app.py` was rewritten into a safer baseline for modern Streamlit and pandas.
- The dependency manifest and README were updated.
- Full runtime verification was not completed because this app depends on local package installation, TensorFlow compatibility, and live webcam access on this machine.
