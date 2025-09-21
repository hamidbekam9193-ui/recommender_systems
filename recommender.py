# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Place Recommender (SVD + Cosine)",
    page_icon="🧭",
    layout="wide"
)
st.title("🧭 Place Recommender (SVD + Cosine Similarity)")
st.caption("Upload trips, reduce dimensions with SVD, and explore similar places or user recommendations.")

# ---------------------------
# Utilities
# ---------------------------
def build_demo_data(n_users=30, n_places=40, seed=42):
    rng = np.random.default_rng(seed)
    users = [f"user_{i}" for i in range(n_users)]
    places = [f"place_{j}" for j in range(n_places)]

    rows = []
    # Simulate that each user visited 5–12 places, with a few popular ones
    popular = set(rng.choice(places, size=max(3, n_places // 8), replace=False))
    for u in users:
        k = rng.integers(5, 13)
        chosen = set(rng.choice(places, size=k, replace=False))
        # Add some popularity bias
        if rng.random() < 0.8:
            chosen.update(rng.choice(list(popular), size=min(len(popular), 2), replace=False))
        for p in chosen:
            rows.append((u, p))
    return pd.DataFrame(rows, columns=["username", "place_slug"])

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

def validate_df(df: pd.DataFrame) -> tuple[bool, str]:
    required = {"username", "place_slug"}
    if not required.issubset(df.columns):
        return False, f"Missing required columns: {sorted(list(required - set(df.columns)))}"
    # Drop NAs and duplicates
    return True, ""

@st.cache_data(show_spinner=False)
def pivot_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # Label-encode to follow the original approach
    le_user = LabelEncoder()
    le_place = LabelEncoder()
    df = df.copy()
    df["username_id"] = le_user.fit_transform(df["username"])
    df["place_slug_id"] = le_place.fit_transform(df["place_slug"])

    # Crosstab (user x place) implicit feedback matrix
    matrix = pd.crosstab(df["username_id"], df["place_slug_id"])
    # Also keep reverse maps
    id2user = {i: u for i, u in enumerate(le_user.classes_)}
    id2place = {i: p for i, p in enumerate(le_place.classes_)}
    user2id = {u: i for i, u in id2user.items()}
    place2id = {p: i for i, p in id2place.items()}
    return matrix, id2user, id2place, user2id, place2id

@st.cache_resource(show_spinner=False)
def fit_svd_and_similarity(matrix: pd.DataFrame, n_components: int, n_iter: int, seed: int):
    if matrix.shape[1] < 2:
        raise ValueError("Not enough places to compute similarities (need at least 2 columns).")
    # Step 2: SVD on place vectors (transpose of user x place)
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=seed)
    matrix_places = svd.fit_transform(matrix.T)  # shape: (n_places, n_components)

    # Step 3: cosine distances → convert to similarities
    dist = cosine_distances(matrix_places)  # shape: (n_places, n_places)
    sim = 1.0 - dist
    np.fill_diagonal(sim, 0.0)  # don't recommend the item itself
    explained = svd.explained_variance_ratio_.sum()
    return sim, explained

def top_k_similar_places(place_id: int, sim_matrix: np.ndarray, k: int = 10):
    scores = sim_matrix[place_id]
    idx = np.argsort(scores)[::-1]
    idx = [i for i in idx if i != place_id]
    idx = idx[:k]
    return idx, scores[idx]

def user_recommendations(user_id: int, matrix: pd.DataFrame, sim_matrix: np.ndarray, k: int = 10):
    # Items the user interacted with
    user_row = matrix.iloc[user_id].values  # occurrences per place
    visited_ids = set(np.where(user_row > 0)[0])

    if len(visited_ids) == 0:
        return [], []

    # Score each candidate place by mean similarity to visited places (simple, fast)
    scores = sim_matrix[:, list(visited_ids)].mean(axis=1)
    # Remove visited
    for v in visited_ids:
        scores[v] = -np.inf

    idx = np.argsort(scores)[::-1][:k]
    vals = scores[idx]
    # Filter out -inf in pathological cases
    mask = ~np.isneginf(vals)
    return idx[mask], vals[mask]

# ---------------------------
# Sidebar / Inputs
# ---------------------------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV with columns: username, place_slug", type=["csv"])
    if uploaded:
        df_raw = load_csv(uploaded)
        st.success("CSV loaded.")
    else:
        st.info("No file? Use demo data.")
        if st.button("Load demo data"):
            df_raw = build_demo_data()
        else:
            df_raw = None

    st.header("Model")
    n_components = st.slider("SVD components (latent factors)", 2, 64, 5, help="As in your original code: default 5.")
    n_iter = st.slider("SVD iterations", 5, 25, 7)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Recommendations")
    k = st.slider("Top-K results", 3, 30, 10)

# ---------------------------
# Main App Logic
# ---------------------------
if df_raw is None:
    st.stop()

ok, msg = validate_df(df_raw)
if not ok:
    st.error(msg)
    st.stop()

# Light cleaning
df = df_raw[["username", "place_slug"]].dropna().drop_duplicates().astype(str)

# Build matrix and encoders
with st.spinner("Building interaction matrix..."):
    matrix, id2user, id2place, user2id, place2id = pivot_interactions(df)

n_users, n_places = matrix.shape
st.subheader("Dataset Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Users", f"{n_users:,}")
c2.metric("Places", f"{n_places:,}")
c3.metric("Interactions", f"{len(df):,}")

# Fit SVD + Similarity
try:
    with st.spinner("Fitting SVD and computing similarities..."):
        sim_matrix, explained = fit_svd_and_similarity(matrix, n_components, n_iter, seed)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.caption(f"Explained variance (SVD): **{explained:.2%}**")

# ---------------------------
# Tabs: Similar Places / User Recs
# ---------------------------
tab1, tab2 = st.tabs(["🔎 Similar Places", "👤 User Recommendations"])

with tab1:
    st.write("Select a place to find the most similar places (by cosine similarity in latent space).")
    place_names = [id2place[i] for i in range(n_places)]
    place_sel = st.selectbox("Pick a place", options=sorted(place_names))
    if place_sel:
        pid = place2id[place_sel]
        idx, scores = top_k_similar_places(pid, sim_matrix, k=k)
        if len(idx) == 0:
            st.warning("Not enough data to compute similarities for this place.")
        else:
            result = pd.DataFrame({
                "place_slug": [id2place[i] for i in idx],
                "similarity": np.round(scores, 6)
            })
            st.dataframe(result, use_container_width=True)

with tab2:
    st.write("Pick a user to get place recommendations they haven’t visited yet.")
    user_names = [id2user[i] for i in range(n_users)]
    user_sel = st.selectbox("Pick a user", options=sorted(user_names))
    if user_sel:
        uid = user2id[user_sel]
        idx, scores = user_recommendations(uid, matrix, sim_matrix, k=k)
        if len(idx) == 0:
            st.warning("No recommendations available (user may have no history or data is too sparse).")
        else:
            visited_ids = set(np.where(matrix.iloc[uid].values > 0)[0])
            rec_df = pd.DataFrame({
                "place_slug": [id2place[i] for i in idx],
                "score": np.round(scores, 6),
                "already_visited": [id2place[i] in {id2place[j] for j in visited_ids} for i in idx]
            })
            rec_df = rec_df[rec_df["already_visited"] == False].drop(columns=["already_visited"])
            if rec_df.empty:
                st.info("User has already visited all highly similar places.")
            else:
                st.dataframe(rec_df, use_container_width=True)

# ---------------------------
# Footer / Tips
# ---------------------------
with st.expander("Implementation details"):
    st.code(
        """# Core steps mirrored from your snippet:
le_user = LabelEncoder()
le_place = LabelEncoder()
df['username_id'] = le_user.fit_transform(df['username'])
df['place_slug_id'] = le_place.fit_transform(df['place_slug'])
matrix = pd.crosstab(df['username_id'], df['place_slug_id'])
svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=seed)
matrix_places = svd.fit_transform(matrix.T)
cosine_distance_matrix_places = cosine_distances(matrix_places)
similarity = 1 - cosine_distance_matrix_places
""",
        language="python",
    )

st.caption("Tip: Increase components if you have many places; decrease if your data is small/sparse.")
