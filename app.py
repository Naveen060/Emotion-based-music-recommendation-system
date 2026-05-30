from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "muse_v3.csv"
MODEL_PATH = BASE_DIR / "model.h5"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

EMOTION_BUCKET_MAP = {
    "Angry": "angry",
    "Disgusted": "angry",
    "Fearful": "fear",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "happy",
}

RECOMMENDATION_SPLITS = {
    1: [30],
    2: [20, 10],
    3: [15, 10, 5],
    4: [10, 9, 8, 3],
    5: [10, 7, 6, 5, 2],
}

cv2.ocl.setUseOpenCL(False)


@st.cache_data(show_spinner=False)
def load_song_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file was not found: {DATASET_PATH.name}")

    df = pd.read_csv(DATASET_PATH)
    required_columns = {
        "lastfm_url",
        "track",
        "number_of_emotion_tags",
        "valence_tags",
        "artist",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    cleaned = df.rename(
        columns={
            "lastfm_url": "link",
            "track": "name",
            "number_of_emotion_tags": "emotional",
            "valence_tags": "pleasant",
        }
    )[["name", "emotional", "pleasant", "link", "artist"]].dropna(subset=["name", "link", "artist"])

    cleaned = cleaned.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)
    chunks = np.array_split(cleaned, 5)

    return {
        "sad": chunks[0].copy(),
        "fear": chunks[1].copy(),
        "angry": chunks[2].copy(),
        "neutral": chunks[3].copy(),
        "happy": chunks[4].copy(),
    }


@st.cache_resource(show_spinner=False)
def load_face_detector():
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise FileNotFoundError(f"Cascade file could not be loaded: {CASCADE_PATH.name}")
    return detector


def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    return model


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights file was not found: {MODEL_PATH.name}")

    model = build_model()
    model.load_weights(str(MODEL_PATH))
    return model


def prioritize_emotions(emotions):
    ordered = []
    for emotion_name, _count in Counter(emotions).most_common():
        normalized = EMOTION_BUCKET_MAP.get(emotion_name)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered[:5]


def recommendation_counts(emotion_count):
    return RECOMMENDATION_SPLITS.get(emotion_count, RECOMMENDATION_SPLITS[5])


def recommend_songs(emotions, song_groups):
    if not emotions:
        return pd.DataFrame(columns=["name", "artist", "link"])

    normalized = prioritize_emotions(emotions)
    counts = recommendation_counts(len(normalized))
    frames = []

    for emotion_name, sample_size in zip(normalized, counts):
        group = song_groups.get(emotion_name)
        if group is None or group.empty:
            continue
        actual_size = min(sample_size, len(group))
        frames.append(group.sample(n=actual_size, replace=False))

    if not frames:
        return pd.DataFrame(columns=["name", "artist", "link"])

    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["name", "artist"]).head(30)


def scan_emotions(face_detector, model, frame_limit=20):
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not access the webcam. Check camera permissions and try again.")

    detected_emotions = []

    try:
        for _ in range(frame_limit):
            ret, frame = capture.read()
            if not ret or frame is None:
                raise RuntimeError("Webcam frames could not be read during emotion scan.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for x, y, w, h in faces:
                roi_gray = gray[y : y + h, x : x + w]
                cropped_img = np.expand_dims(
                    np.expand_dims(cv2.resize(roi_gray, (48, 48)), axis=-1),
                    axis=0,
                )
                prediction = model.predict(cropped_img, verbose=0)
                emotion_label = EMOTION_LABELS[int(np.argmax(prediction))]
                detected_emotions.append(emotion_label)

                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    emotion_label,
                    (x + 20, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Emotion Scan", cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord("x"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return detected_emotions


def render_recommendations(recommendations):
    st.write("")
    st.markdown(
        "<h5 style='text-align: center; color: grey;'><b>Recommended songs with artist names</b></h5>",
        unsafe_allow_html=True,
    )
    st.write("---------------------------------------------------------------------------------------------------------------------")

    if recommendations.empty:
        st.info("No recommendations are available yet. Run an emotion scan first.")
        return

    for index, row in recommendations.reset_index(drop=True).iterrows():
        st.markdown(
            f"<h4 style='text-align: center;'><a href='{row['link']}'>{index + 1}. {row['name']}</a></h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h5 style='text-align: center; color: grey;'><i>{row['artist']}</i></h5>",
            unsafe_allow_html=True,
        )
        st.write("---------------------------------------------------------------------------------------------------------------------")


def main():
    st.set_page_config(page_title="Emotion Based Music Recommendation", layout="centered")
    st.markdown(
        "<h2 style='text-align: center; color: white;'><b>Emotion based music recommendation</b></h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h5 style='text-align: center; color: grey;'><b>Click on a recommended song title to open it</b></h5>",
        unsafe_allow_html=True,
    )

    if "detected_emotions" not in st.session_state:
        st.session_state.detected_emotions = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = pd.DataFrame(columns=["name", "artist", "link"])

    try:
        song_groups = load_song_dataset()
        face_detector = load_face_detector()
        model = load_emotion_model()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Scan Emotion"):
            try:
                detected_emotions = scan_emotions(face_detector, model)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.session_state.detected_emotions = detected_emotions
                st.session_state.recommendations = recommend_songs(detected_emotions, song_groups)

    if st.session_state.detected_emotions:
        summary = ", ".join(prioritize_emotions(st.session_state.detected_emotions))
        st.caption(f"Detected emotion priority: {summary}")

    render_recommendations(st.session_state.recommendations)


if __name__ == "__main__":
    main()
