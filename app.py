import datetime
import re
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

# ==========================================
# 1. SƏHFƏ VƏ MODEL QURULUSU (SETUP)
# ==========================================

st.set_page_config(
    page_title="Real-Time Topic Detection", page_icon="🐦", layout="wide"
)

st.title("🐦 Real-Time Twitter Topic Detection Dashboard")
st.markdown("Canlı tvit axınında mövzuların real vaxtda təsnifatı və analizi")


# Modeli önyaddaşa alırıq ki, təkrarlanan dövrdə yenidən yüklənməsin
@st.cache_resource
def load_nlp_model():
    # Sürətli işləməsi üçün distilbart modeli tövsiyə olunur
    return pipeline(
        "zero-shot-classification", model="valhalla/distilbart-mnli-12-3"
    )


classifier = load_nlp_model()

# ==========================================
# 2. İSTİFADƏÇİ PANATELİ (SIDEBAR)
# ==========================================

st.sidebar.header("⚙️ Axın Parametrləri")

# Sürət tənzimləməsi
stream_speed = st.sidebar.slider(
    "Axın sürəti (saniyə/tvit):",
    min_value=0.2,
    max_value=3.0,
    value=1.0,
    step=0.2,
)

# Mövzuların dinamik seçimi (Bonus Tələbi)
default_topics = ["Sports", "Politics", "Tech", "Entertainment"]
user_topics = st.sidebar.text_input(
    "Mövzuları vergüllə daxil edin:", value=", ".join(default_topics)
)
topics_list = [t.strip() for t in user_topics.split(",") if t.strip()]

start_button = st.sidebar.button("🚀 Axını Başlat", type="primary")

# ==========================================
# 3. KÖMƏKÇİ FUNKSİYALAR
# ==========================================


def clean_tweet(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def load_sample_data():
    try:
        # Sentiment140 faylını oxumağa cəhd edir
        cols = ["target", "id", "date", "flag", "user", "text"]
        df = pd.read_csv(
            "training.1600000.processed.noemoticon.csv",
            encoding="latin-1",
            names=cols,
        )
        df["clean_text"] = df["text"].apply(clean_tweet)
        return df[df["clean_text"].str.len() > 3]["clean_text"].tolist()
    except FileNotFoundError:
        # CSV tapılmazsa simulyasiya üçün hazır tvitlər
        return [
            "Apple just released their new M3 MacBook Pro with crazy fast specs!",
            "What an incredible game yesterday! The last-minute goal was insane.",
            "The upcoming elections are going to reshape international relations.",
            "Cybersecurity threat detected in modern cloud architectures.",
            "Movie theaters are witnessing record box office numbers this weekend.",
            "AI technology is transforming software engineering at a rapid pace.",
            "The referee made a very controversial decision during the final match.",
            "New policy changes in parliament raised debates among citizens.",
        ] * 20


tweets_data = load_sample_data()

# ==========================================
# 4. DASHBOARD KONTEYNERLƏRİ (PLACEHOLDERS)
# ==========================================

metric_col1, metric_col2, metric_col3 = st.columns(3)

metrics_p1 = metric_col1.empty()
metrics_p2 = metric_col2.empty()
metrics_p3 = metric_col3.empty()

st.divider()

col_left, col_right = st.columns(2)
bar_chart_p = col_left.empty()
line_chart_p = col_right.empty()

st.divider()
st.subheader("📜 Canlı Tvit Axını Paneli")
table_p = st.empty()

# ==========================================
# 5. REAL-TIME AXIN DÖVRÜ (CANLI İCRA)
# ==========================================

if start_button:
    # Məlumatları toplamaq üçün boş DataFrame
    processed_records = []

    for i, raw_text in enumerate(tweets_data):
        # Model təsnifatı
        result = classifier(raw_text, topics_list)
        top_topic = result["labels"][0]
        confidence = round(result["scores"][0] * 100, 1)
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # Nəticəni saxlayırıq
        processed_records.append(
            {
                "Timestamp": current_time,
                "Tweet": raw_text,
                "Topic": top_topic,
                "Confidence (%)": confidence,
            }
        )

        df_processed = pd.DataFrame(processed_records)

        # --- KPI Metrikalarının Yenilənməsi ---
        metrics_p1.metric("Ümumi Emal Olunan Tvit", len(df_processed))
        top_active_topic = df_processed["Topic"].mode()[0]
        metrics_p2.metric("Ən Çox Təkrar Olunan Mövzu", top_active_topic)
        metrics_p3.metric("Son Tvitin Mövzusu", top_topic)

        # --- 1. Bar Chart (Mövzu Tezliyi) ---
        topic_counts = df_processed["Topic"].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]

        fig_bar = px.bar(
            topic_counts,
            x="Topic",
            y="Count",
            color="Topic",
            title="Mövzu Sayı Paylanması",
            text="Count",
        )
        fig_bar.update_layout(showlegend=False, height=350)
        bar_chart_p.plotly_chart(fig_bar, use_container_width=True)

        # --- 2. Line Chart (Zaman üzrə mövzu trendi) ---
        trend_df = (
            df_processed.groupby(["Timestamp", "Topic"])
            .size()
            .unstack(fill_value=0)
            .cumsum()
        )
        fig_line = px.line(
            trend_df, title="Mövzuların Vaxta Görə Artım Trendi (Kumulyativ)"
        )
        fig_line.update_layout(
            xaxis_title="Vaxt", yaxis_title="Kumulyativ Say", height=350
        )
        line_chart_p.plotly_chart(fig_line, use_container_width=True)

        # --- 3. Canlı Cədvəl Logu ---
        table_p.dataframe(
            df_processed.tail(7)[
                ["Timestamp", "Topic", "Confidence (%)", "Tweet"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        time.sleep(stream_speed)
else:
    st.info("Axını başlatmaq üçün sol paneldən '🚀 Axını Başlat' düyməsini sıxın.")