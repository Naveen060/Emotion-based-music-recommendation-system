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

    cleaned = (
        df.rename(
            columns={
                "lastfm_url": "link",
                "track": "name",
                "number_of_emotion_tags": "emotional",
                "valence_tags": "pleasant",
            }
        )[["name", "emotional", "pleasant", "link", "artist"]]
        .dropna(subset=["name", "link", "artist"])
        .sort_values(by=["emotional", "pleasant"])
        .reset_index(drop=True)
    )

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


def normalize_emotions(emotions):
    mapped = [EMOTION_BUCKET_MAP.get(emotion) for emotion in emotions if EMOTION_BUCKET_MAP.get(emotion)]
    ordered = []
    for emotion_name, _count in Counter(mapped).most_common():
        if emotion_name not in ordered:
            ordered.append(emotion_name)
    return ordered


def recommendation_plan(emotions, recommendation_count):
    unique_count = max(1, len(emotions))
    weights = {
        1: [1.0],
        2: [0.65, 0.35],
        3: [0.5, 0.3, 0.2],
        4: [0.4, 0.25, 0.2, 0.15],
        5: [0.34, 0.23, 0.18, 0.15, 0.10],
    }[min(unique_count, 5)]

    counts = [max(1, round(recommendation_count * weight)) for weight in weights]
    while sum(counts) > recommendation_count:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < recommendation_count:
        counts[counts.index(min(counts))] += 1
    return counts


def recommend_songs(emotions, song_groups, recommendation_count):
    normalized = normalize_emotions(emotions)
    if not normalized:
        return pd.DataFrame(columns=["name", "artist", "link", "bucket"])

    counts = recommendation_plan(normalized, recommendation_count)
    frames = []

    for emotion_name, sample_size in zip(normalized, counts):
        group = song_groups.get(emotion_name)
        if group is None or group.empty:
            continue
        actual_size = min(sample_size, len(group))
        sampled = group.sample(n=actual_size, replace=False).copy()
        sampled["bucket"] = emotion_name
        frames.append(sampled)

    if not frames:
        return pd.DataFrame(columns=["name", "artist", "link", "bucket"])

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["name", "artist"])
        .head(recommendation_count)
        .reset_index(drop=True)
    )


def regenerate_recommendations(song_groups, recommendation_count):
    if not st.session_state.detected_emotions:
        return pd.DataFrame(columns=["name", "artist", "link", "bucket"])
    return recommend_songs(st.session_state.detected_emotions, song_groups, recommendation_count)


def scan_emotions(face_detector, model, frame_limit):
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

                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 132, 92), 2)
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


def recommendation_download_frame(recommendations):
    if recommendations.empty:
        return recommendations
    return recommendations[["name", "artist", "bucket", "link"]]


def render_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 132, 92, 0.16), transparent 30%),
                    radial-gradient(circle at top right, rgba(73, 190, 255, 0.12), transparent 20%),
                    linear-gradient(160deg, #07111b 0%, #101f2c 55%, #162636 100%);
            }
            .hero-panel {
                padding: 1.4rem 1.6rem;
                border-radius: 24px;
                background: rgba(11, 20, 31, 0.82);
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 18px 48px rgba(0,0,0,0.28);
                margin-bottom: 1.2rem;
            }
            .hero-title {
                font-size: 2.2rem;
                margin-bottom: 0.3rem;
            }
            .hero-copy {
                color: #b9c6d2;
                line-height: 1.6;
                margin: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Emotion Based Music Recommendation", layout="wide")
    render_styles()

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">Emotion Based Music Recommendation</div>
            <p class="hero-copy">
                Scan facial emotion from a webcam feed, rank the dominant mood signals, and generate a playlist suggestion
                from the bundled music dataset. This refreshed version adds better controls, reusable recommendations,
                and export-ready results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "detected_emotions" not in st.session_state:
        st.session_state.detected_emotions = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = pd.DataFrame(columns=["name", "artist", "link", "bucket"])

    try:
        song_groups = load_song_dataset()
        face_detector = load_face_detector()
        model = load_emotion_model()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    with st.sidebar:
        st.header("Controls")
        frame_limit = st.slider("Frames to scan", min_value=10, max_value=40, value=20, step=5)
        recommendation_count = st.slider("Songs to recommend", min_value=10, max_value=30, value=20, step=5)
        st.caption("Press `x` in the OpenCV window if you want to stop scanning early.")

    if st.button("Scan Emotion", type="primary"):
        try:
            detected_emotions = scan_emotions(face_detector, model, frame_limit=frame_limit)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.session_state.detected_emotions = detected_emotions
            st.session_state.recommendations = recommend_songs(
                detected_emotions,
                song_groups,
                recommendation_count=recommendation_count,
            )

    if st.session_state.detected_emotions and st.button("Refresh Playlist Without Rescan"):
        st.session_state.recommendations = regenerate_recommendations(song_groups, recommendation_count)

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Captured Emotion Frames", len(st.session_state.detected_emotions))
    metric_b.metric("Unique Mood Buckets", len(normalize_emotions(st.session_state.detected_emotions)))
    metric_c.metric("Recommended Songs", len(st.session_state.recommendations))

    if st.session_state.detected_emotions:
        emotion_counts = Counter(st.session_state.detected_emotions)
        summary = ", ".join(
            f"{emotion} ({count})"
            for emotion, count in emotion_counts.most_common()
        )
        st.info(f"Detected emotion summary: {summary}")

        breakdown_frame = pd.DataFrame(
            [{"emotion": emotion, "count": count} for emotion, count in emotion_counts.most_common()]
        )
        left, right = st.columns([1.2, 2])
        with left:
            st.subheader("Emotion Breakdown")
            st.dataframe(breakdown_frame, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Recommended Playlist")
            if st.session_state.recommendations.empty:
                st.warning("No recommendations could be generated from the detected emotion frames.")
            else:
                bucket_filter = st.selectbox(
                    "Filter playlist by mood bucket",
                    ["all"] + sorted(st.session_state.recommendations["bucket"].unique().tolist()),
                )
                playlist = (
                    st.session_state.recommendations
                    if bucket_filter == "all"
                    else st.session_state.recommendations[st.session_state.recommendations["bucket"] == bucket_filter]
                )
                for index, row in playlist.reset_index(drop=True).iterrows():
                    st.markdown(
                        f"**{index + 1}. [{row['name']}]({row['link']})**  \n"
                        f"{row['artist']}  \n"
                        f"`bucket: {row['bucket']}`"
                    )
                    st.divider()

                csv_bytes = recommendation_download_frame(playlist).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Recommendations CSV",
                    data=csv_bytes,
                    file_name="emotion_playlist.csv",
                    mime="text/csv",
                )
    else:
        st.caption("Run a scan to generate mood-based recommendations.")


if __name__ == "__main__":
    main()
