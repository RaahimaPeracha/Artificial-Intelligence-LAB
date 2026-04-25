import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0a;
    color: #f0f0f0;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main { background-color: #0a0a0a; }

section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #1db954;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1db954 0%, #0a4d22 50%, #0a0a0a 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.1rem;
    color: #b3ffcb;
    margin: 0;
    font-weight: 300;
}

/* Slider labels */
.slider-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #1db954;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 2px;
}

/* Song card */
.song-card {
    background: #161616;
    border: 1px solid #222;
    border-left: 4px solid #1db954;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.song-card:hover { border-left-color: #3dff7a; }

.song-rank {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #1db954;
    line-height: 1;
}
.song-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 4px 0 2px 0;
}
.song-artist { font-size: 0.9rem; color: #aaa; }
.song-meta   { font-size: 0.8rem; color: #666; margin-top: 6px; }

.match-bar-bg {
    background: #222;
    border-radius: 999px;
    height: 6px;
    margin-top: 10px;
}
.match-bar-fill {
    background: linear-gradient(90deg, #1db954, #3dff7a);
    border-radius: 999px;
    height: 6px;
}
.match-pct {
    font-size: 0.78rem;
    color: #1db954;
    font-weight: 600;
    margin-top: 4px;
}

/* Genre badge */
.genre-badge {
    display: inline-block;
    background: #1a3a24;
    color: #1db954;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Predict button */
div.stButton > button {
    background: #1db954;
    color: #000;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 999px;
    padding: 12px 40px;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
}
div.stButton > button:hover {
    background: #3dff7a;
    transform: scale(1.02);
}

/* Predicted genre box */
.genre-result {
    background: linear-gradient(135deg, #1a3a24, #0d1f14);
    border: 1px solid #1db954;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 24px;
}
.genre-result p { margin: 0; color: #b3ffcb; font-size: 0.9rem; }
.genre-result h2 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    color: #1db954;
    margin: 4px 0 0 0;
    text-transform: capitalize;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 24px 0 12px 0;
    border-bottom: 1px solid #222;
    padding-bottom: 8px;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model & data (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('spotify.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('spotify-tracks-dataset-detailed.csv')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_resource
def build_encoder(df):
    TOP = 20
    top_genres = df['track_genre'].value_counts().head(TOP).index.tolist()
    ml_df = df[df['track_genre'].isin(top_genres)].copy()
    le = LabelEncoder()
    le.fit(ml_df['track_genre'])
    return le

model = load_model()
df    = load_data()
le    = build_encoder(df)

FEATURES = [
    "popularity", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature", "explicit", "duration_min"
]

MATCH_FEATURES = [
    "popularity", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature", "explicit", "duration_ms"
]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎵 Spotify Recommender</h1>
    <p>Tune your preferences — get your perfect songs instantly.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Your Preferences")
    st.markdown("---")

    st.markdown('<p class="section-title">Mood & Energy</p>', unsafe_allow_html=True)
    popularity   = st.slider("Popularity (0–100)",       0,    100,  70)
    valence      = st.slider("Valence / Happiness",      0.0,  1.0,  0.6, 0.01)
    energy       = st.slider("Energy",                   0.0,  1.0,  0.7, 0.01)
    danceability = st.slider("Danceability",             0.0,  1.0,  0.7, 0.01)

    st.markdown('<p class="section-title">Sound</p>', unsafe_allow_html=True)
    acousticness     = st.slider("Acousticness",         0.0,  1.0,  0.2, 0.01)
    instrumentalness = st.slider("Instrumentalness",     0.0,  1.0,  0.0, 0.01)
    speechiness      = st.slider("Speechiness",          0.0,  1.0,  0.05, 0.01)
    liveness         = st.slider("Liveness",             0.0,  1.0,  0.1, 0.01)

    st.markdown('<p class="section-title">Technical</p>', unsafe_allow_html=True)
    tempo          = st.slider("Tempo (BPM)",            0.0,  243.0, 120.0, 1.0)
    loudness       = st.slider("Loudness (dB)",         -60.0, 0.0,  -6.0,  0.5)
    key            = st.selectbox("Key (0–11)",          list(range(12)), index=5)
    mode           = st.selectbox("Mode",                [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
    time_signature = st.selectbox("Time Signature",      [3, 4, 5], index=1)
    explicit       = st.selectbox("Explicit",            [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    duration_ms    = st.number_input("Duration (ms)",    min_value=30000, max_value=600000, value=200000, step=1000)

    st.markdown("---")
    top_n    = st.slider("Number of recommendations", 1, 10, 5)
    predict  = st.button("🎵 Find My Songs")

# ── Main content ──────────────────────────────────────────────────────────────
if predict:
    track_data = {
        "popularity":        popularity,
        "danceability":      int(danceability),
        "energy":            int(energy),
        "key":               key,
        "loudness":          int(loudness),
        "mode":              mode,
        "speechiness":       int(speechiness),
        "acousticness":      int(acousticness),
        "instrumentalness":  int(instrumentalness),
        "liveness":          int(liveness),
        "valence":           int(valence),
        "tempo":             int(tempo),
        "time_signature":    time_signature,
        "explicit":          explicit,
        "duration_min":      duration_ms,
    }

    # ── Model predicts genre ──────────────────────────────────────────────────
    X = np.array([[track_data[f] for f in FEATURES]])
    pred_encoded = model.predict(X)[0]
    pred_genre   = le.inverse_transform([pred_encoded])[0]

    # ── Show predicted genre ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="genre-result">
        <p>🤖 Model predicted genre</p>
        <h2>{pred_genre}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Cosine similarity within predicted genre ──────────────────────────────
    genre_df = df[df['track_genre'] == pred_genre].copy()
    genre_df.reset_index(drop=True, inplace=True)

    if genre_df.empty:
        st.warning("No songs found for the predicted genre. Try adjusting your inputs.")
    else:
        scaler       = MinMaxScaler()
        genre_scaled = scaler.fit_transform(genre_df[MATCH_FEATURES])

        user_input = {
            "popularity":        track_data["popularity"],
            "danceability":      track_data["danceability"],
            "energy":            track_data["energy"],
            "key":               track_data["key"],
            "loudness":          track_data["loudness"],
            "mode":              track_data["mode"],
            "speechiness":       track_data["speechiness"],
            "acousticness":      track_data["acousticness"],
            "instrumentalness":  track_data["instrumentalness"],
            "liveness":          track_data["liveness"],
            "valence":           track_data["valence"],
            "tempo":             track_data["tempo"],
            "time_signature":    track_data["time_signature"],
            "explicit":          track_data["explicit"],
            "duration_ms":       track_data["duration_min"],
        }

        user_vec     = scaler.transform(pd.DataFrame([user_input]))
        sims         = cosine_similarity(user_vec, genre_scaled)[0]
        top_idx      = np.argsort(sims)[::-1][:top_n]

        st.markdown(f'<p class="section-title">🎶 Top {top_n} Songs in "{pred_genre}"</p>', unsafe_allow_html=True)

        for rank, i in enumerate(top_idx, start=1):
            row      = genre_df.iloc[i]
            match    = sims[i] * 100
            bar_w    = f"{match:.1f}%"
            explicit_tag = "🅴 Explicit" if row['explicit'] else ""

            st.markdown(f"""
            <div class="song-card">
                <div style="display:flex; align-items:flex-start; gap:20px;">
                    <div class="song-rank">#{rank}</div>
                    <div style="flex:1;">
                        <div class="song-title">{row['track_name']}</div>
                        <div class="song-artist">{row['artists']}</div>
                        <div class="song-meta">
                            <span class="genre-badge">{row['track_genre']}</span>
                            &nbsp;⭐ {row['popularity']}/100
                            &nbsp;{explicit_tag}
                        </div>
                        <div class="match-bar-bg">
                            <div class="match-bar-fill" style="width:{bar_w};"></div>
                        </div>
                        <div class="match-pct">{match:.1f}% match</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Placeholder state ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; color: #444;">
        <div style="font-size: 4rem;">🎧</div>
        <p style="font-family:'Syne',sans-serif; font-size:1.3rem; color:#666; margin-top:16px;">
            Adjust the sliders on the left and hit <span style="color:#1db954;">Find My Songs</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
