"""
Streamlit Web Application for Sentiment Analysis on Movie Reviews
Interactive NLP & Text Classification Explorer for Introduction to AI
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import explain_preprocessing_steps, preprocess_text
from src.llm_analyzer import analyze_with_llm, get_ml_vs_llm_comparison_matrix

# Page Configuration
st.set_page_config(
    page_title="AI Sentiment Analysis | Movie Reviews NLP",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode Glassmorphism & Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Header Gradient */
    .hero-title {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Badges */
    .badge-pos {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.35);
    }
    .badge-neg {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.35);
    }
    
    /* Pipeline Step Box */
    .step-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366F1;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LR_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
NB_PATH = os.path.join(MODELS_DIR, "naive_bayes.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "top_features.json")


@st.cache_resource
def load_models():
    """Load vectorizer and classifiers from disk with caching."""
    if not (os.path.exists(VEC_PATH) and os.path.exists(LR_PATH)):
        return None, None, None, None, None
    try:
        vec = joblib.load(VEC_PATH)
        lr = joblib.load(LR_PATH)
        nb = joblib.load(NB_PATH) if os.path.exists(NB_PATH) else None
        
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                
        features = {}
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                features = json.load(f)
                
        return vec, lr, nb, metrics, features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None


# Presets for quick exploration
PRESET_REVIEWS = {
    "🌟 Blockbuster Masterpiece (Tích cực mạnh)": (
        "An absolute masterpiece of modern cinema! The cinematography was breathtaking, "
        "the musical score was profoundly moving, and the lead actor gave the performance of a lifetime. "
        "I was on the edge of my seat from start to finish. Highly recommended!"
    ),
    "💣 Total Disaster (Tiêu cực mạnh)": (
        "An utterly unwatchable disaster and a colossal waste of time. The script is painfully dull, "
        "the dialogue is amateurish, and the plot makes zero sense. Do not waste your money on this garbage."
    ),
    "🎭 Tricky Nuance & Negation (Nghịch lý & Phủ định)": (
        "I walked into the theater expecting a complete failure, but surprisingly, it was not bad at all. "
        "While it is not a masterpiece, the humor was genuinely charming and I did not feel bored for a second."
    ),
    "⚡ Subtle Irony / Mixed Review (Hỗn hợp phức tạp)": (
        "The visual special effects were brilliant and the production design was gorgeous, "
        "yet the atrocious pacing and paper-thin characters completely ruined the emotional core."
    )
}

# Header Section
st.markdown('<div class="hero-title">🎬 Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Hệ thống Phân loại Cảm xúc Đánh giá Phim • Natural Language Processing (NLP) & Machine Learning Pipeline</div>', 
    unsafe_allow_html=True
)

# Load artifacts
vectorizer, lr_model, nb_model, evaluation_metrics, top_features_dict = load_models()

# Sidebar: Configuration and Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2809/2809590.png", width=64)
    st.title("Settings & Stack")
    st.markdown("**Course**: Introduction to AI / Nhập môn Trí tuệ Nhân tạo")
    st.markdown("**Dataset**: IMDb 50,000 Movie Reviews")
    
    st.divider()
    st.subheader("Model Selection")
    selected_model_name = st.radio(
        "Choose Classifier:",
        ["Logistic Regression", "Multinomial Naive Bayes"],
        index=0,
        help="Logistic Regression uses linear hyperplane with sigmoid activation. Naive Bayes applies Bayes' Theorem."
    )
    
    st.divider()
    st.subheader("Model Artifact Status")
    if vectorizer and lr_model:
        st.success("✅ Models Loaded from Disk")
        st.caption(f"Vocabulary Size: **{len(vectorizer.get_feature_names_out()):,} n-grams**")
    else:
        st.warning("⚠️ Baseline models not yet trained on disk.")
        if st.button("⚡ Run Quick Training Baseline", type="primary"):
            with st.spinner("Training baseline model (this takes ~15-20s)..."):
                from src.train import train_and_evaluate
                train_and_evaluate(sample_size=10000, max_features=8000)
                st.cache_resource.clear()
                st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Inference & Pipeline",
    "📊 Model Benchmark & Metrics",
    "📐 NLP Math & Foundations",
    "🤖 LLM Zero-Shot Extension"
])

# ==========================================
# TAB 1: LIVE INFERENCE & PIPELINE EXPLORER
# ==========================================
with tab1:
    st.subheader("Interactive Sentiment Classifier & Pipeline Inspector")
    st.markdown(
        "Type or select a movie review below to inspect how raw text is preprocessed, "
        "transformed into mathematical TF-IDF vector space, and classified."
    )
    
    # Preset selection
    col_preset, col_clear = st.columns([3, 1])
    with col_preset:
        preset_choice = st.selectbox("🎯 Load Sample Preset Review:", ["-- Custom Input --"] + list(PRESET_REVIEWS.keys()))
    
    default_text = PRESET_REVIEWS.get(preset_choice, "") if preset_choice != "-- Custom Input --" else ""
    if not default_text:
        default_text = (
            "An absolute triumph! The direction is visionary, the acting is top notch, "
            "and the emotional payoff at the climax was unforgettable. A must watch masterpiece!"
        )
    
    user_review = st.text_area(
        "Enter English Movie Review:",
        value=default_text,
        height=130,
        placeholder="Type a film review here..."
    )
    
    col_btn, col_algo = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("🔍 Predict Sentiment", type="primary", use_container_width=True)
        
    if user_review.strip():
        # Step-by-step pipeline explanation
        prep_steps = explain_preprocessing_steps(user_review)
        cleaned_str = prep_steps["final_text"]
        
        # Inference
        active_model = lr_model if selected_model_name == "Logistic Regression" else nb_model
        
        if vectorizer is not None and active_model is not None:
            # Transform to TF-IDF
            tfidf_vec = vectorizer.transform([cleaned_str])
            prob = active_model.predict_proba(tfidf_vec)[0]
            pred_class = active_model.predict(tfidf_vec)[0]
            
            pos_prob = prob[1]
            neg_prob = prob[0]
            is_pos = pred_class == 1
            
            # --- RESULT DISPLAY ---
            st.markdown("---")
            res_col1, res_col2 = st.columns([1.2, 2])
            
            with res_col1:
                st.markdown("#### Prediction Result")
                if is_pos:
                    st.markdown(
                        f'<div class="badge-pos">👍 POSITIVE ({pos_prob*100:.1f}%)</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="badge-neg">👎 NEGATIVE ({neg_prob*100:.1f}%)</div>', 
                        unsafe_allow_html=True
                    )
                st.write("")
                st.caption(f"Algorithm: **{selected_model_name}**")
                st.caption(f"Decision Boundary: **0.50**")
                
            with res_col2:
                st.markdown("#### Confidence Probability Distribution")
                prob_df = pd.DataFrame({
                    "Sentiment": ["Negative (Tiêu cực)", "Positive (Tích cực)"],
                    "Probability": [neg_prob, pos_prob],
                    "Color": ["#EF4444", "#10B981"]
                })
                
                chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
                    x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                    y=alt.Y('Sentiment:N', sort=None, title=None),
                    color=alt.Color('Sentiment:N', scale=alt.Scale(
                        domain=["Negative (Tiêu cực)", "Positive (Tích cực)"],
                        range=["#EF4444", "#10B981"]
                    ), legend=None),
                    tooltip=['Sentiment', alt.Tooltip('Probability:Q', format='.2%')]
                ).properties(height=110)
                st.altair_chart(chart, use_container_width=True)

            # --- MATHEMATICAL PIPELINE BREAKDOWN ---
            st.markdown("---")
            st.markdown("### 🔬 Step-by-Step Data Transformation Pipeline")
            st.markdown("Visualizing exactly how the data evolves from raw characters to a mathematical classification decision:")
            
            with st.expander("📌 Step 1: Raw Text & Contraction Normalization", expanded=True):
                st.write("**Raw Input Review:**")
                st.code(prep_steps["raw_text"], language="text")
                st.write("**After HTML Stripping & Contraction Expansion (preserving negation):**")
                st.code(prep_steps["expanded_contractions"], language="text")
                
            with st.expander("✂️ Step 2: Stopwords Filtering & Lemmatization", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Words", prep_steps["tokens_count_initial"])
                c2.metric("Retained Tokens", prep_steps["tokens_count_retained"])
                c3.metric("Noise Reduction", f"{prep_steps['reduction_percentage']}%")
                
                st.write("**Filtered Stopwords (Noise Removed):**")
                st.caption(", ".join(set(prep_steps["removed_stopwords"])) if prep_steps["removed_stopwords"] else "None")
                
                st.write("**Lemmatized Words (Base Dictionary Roots):**")
                st.code(" ".join(prep_steps["lemmatized_tokens"]), language="text")
                st.caption("Notice how plural nouns become singular and conjugated verbs map to base roots while negations ('not', 'never') are deliberately preserved.")
                
            with st.expander("🧮 Step 3: TF-IDF Vector Space Representation (Trích xuất Vector)", expanded=True):
                feature_names = vectorizer.get_feature_names_out()
                nonzero_indices = tfidf_vec.nonzero()[1]
                
                if len(nonzero_indices) > 0:
                    active_features = [
                        {
                            "Feature (n-gram)": feature_names[idx],
                            "TF-IDF Weight": round(float(tfidf_vec[0, idx]), 4),
                            "IDF": round(float(vectorizer.idf_[idx]), 4)
                        }
                        for idx in nonzero_indices
                    ]
                    active_df = pd.DataFrame(active_features).sort_values(by="TF-IDF Weight", ascending=False)
                    
                    st.markdown(f"**This review activated {len(nonzero_indices)} distinct vocabulary features in the 10,000-dimensional sparse vector space:**")
                    
                    col_t1, col_t2 = st.columns([1.5, 1])
                    with col_t1:
                        # Bar chart of top TF-IDF weights
                        top_plot_df = active_df.head(10)
                        tfidf_chart = alt.Chart(top_plot_df).mark_bar(color='#6366F1').encode(
                            x=alt.X('TF-IDF Weight:Q', title='TF-IDF Score'),
                            y=alt.Y('Feature (n-gram):N', sort='-x', title='Active Term'),
                            tooltip=['Feature (n-gram)', 'TF-IDF Weight', 'IDF']
                        ).properties(height=240, title="Top Active Feature Weights in this Review")
                        st.altair_chart(tfidf_chart, use_container_width=True)
                    with col_t2:
                        st.dataframe(active_df, height=240, use_container_width=True)
                else:
                    st.info("No vocabulary words matched the trained dictionary (Out-of-Vocabulary).")
                    
            with st.expander("🎯 Step 4: Mathematical Decision Calculation", expanded=True):
                if selected_model_name == "Logistic Regression":
                    coefs = active_model.coef_[0]
                    intercept = float(active_model.intercept_[0])
                    
                    # Compute feature contributions w_i * x_i
                    contributions = []
                    for idx in nonzero_indices:
                        feat = feature_names[idx]
                        w = coefs[idx]
                        x = tfidf_vec[0, idx]
                        contribution = w * x
                        contributions.append({
                            "Feature": feat,
                            "TF-IDF (x)": round(float(x), 4),
                            "Weight (w)": round(float(w), 4),
                            "Contribution (w * x)": round(float(contribution), 4),
                            "Polarity": "Positive" if w > 0 else "Negative"
                        })
                    
                    contrib_df = pd.DataFrame(contributions).sort_values(by="Contribution (w * x)", ascending=False)
                    total_z = float(np.dot(tfidf_vec.toarray()[0], coefs) + intercept)
                    sigmoid_val = 1 / (1 + np.exp(-total_z))
                    
                    st.markdown("#### Logistic Regression Sigmoid Formula:")
                    st.latex(r"z = \sum_{i=1}^V w_i x_i + b \quad \longrightarrow \quad P(\text{Positive}|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Bias / Intercept (b)", f"{intercept:.4f}")
                    m2.metric("Logit Score (z)", f"{total_z:.4f}")
                    m3.metric("Final Probability σ(z)", f"{sigmoid_val:.4%}")
                    
                    st.markdown("**Feature Contributions to Decision:**")
                    st.dataframe(contrib_df, height=200, use_container_width=True)
                else:
                    st.markdown("#### Multinomial Naive Bayes Decision Rule:")
                    st.latex(r"P(C|D) \propto P(C) \prod_{i=1}^n P(w_i|C)")
                    st.markdown("Log-space computation applies Laplace smoothing $\\alpha=1$ to avoid zero-probabilities:")
                    st.latex(r"\log P(C|D) = \log P(C) + \sum_{i=1}^n x_i \log P(w_i|C)")
        else:
            st.info("💡 Train or load models via sidebar to enable live classification.")

# ==========================================
# TAB 2: MODEL BENCHMARK & METRICS
# ==========================================
with tab2:
    st.subheader("Model Evaluation & Algorithm Comparison")
    st.markdown(
        "Comparative performance analysis on the IMDb 50,000 dataset (stratified 80% Train / 20% Test split). "
        "Evaluating **Logistic Regression** vs **Multinomial Naive Bayes**."
    )
    
    if evaluation_metrics:
        # Side-by-side metric cards
        lr_m = evaluation_metrics.get("Logistic Regression", {})
        nb_m = evaluation_metrics.get("Multinomial Naive Bayes", {})
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### 🔷 Logistic Regression")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{lr_m.get('accuracy', 0)*100:.2f}%")
            c2.metric("Precision", f"{lr_m.get('precision', 0):.4f}")
            c3.metric("Recall", f"{lr_m.get('recall', 0):.4f}")
            c4.metric("F1-Score", f"{lr_m.get('f1_score', 0):.4f}")
            st.caption(f"⚡ Inference Latency: **{lr_m.get('inference_time_seconds', 0)*1000/lr_m.get('test_samples', 1):.2f} ms / review**")
            
        with col_m2:
            st.markdown("### 🔶 Multinomial Naive Bayes")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{nb_m.get('accuracy', 0)*100:.2f}%")
            c2.metric("Precision", f"{nb_m.get('precision', 0):.4f}")
            c3.metric("Recall", f"{nb_m.get('recall', 0):.4f}")
            c4.metric("F1-Score", f"{nb_m.get('f1_score', 0):.4f}")
            st.caption(f"⚡ Inference Latency: **{nb_m.get('inference_time_seconds', 0)*1000/nb_m.get('test_samples', 1):.2f} ms / review**")
            
        st.divider()
        
        # Confusion Matrices
        st.subheader("Confusion Matrices (Ma trận Nhầm lẫn)")
        cm_col1, cm_col2 = st.columns(2)
        
        def plot_cm(cm_data, title):
            data = [
                {"Actual": "Negative", "Predicted": "Negative", "Count": cm_data[0][0]},
                {"Actual": "Negative", "Predicted": "Positive", "Count": cm_data[0][1]},
                {"Actual": "Positive", "Predicted": "Negative", "Count": cm_data[1][0]},
                {"Actual": "Positive", "Predicted": "Positive", "Count": cm_data[1][1]},
            ]
            df_cm = pd.DataFrame(data)
            base = alt.Chart(df_cm).encode(
                x='Predicted:N',
                y='Actual:N'
            )
            heatmap = base.mark_rect().encode(
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='purples'), legend=None)
            )
            text = base.mark_text(baseline='middle', fontSize=14, fontWeight='bold').encode(
                text='Count:Q',
                color=alt.condition(alt.datum.Count > df_cm['Count'].max()/2, alt.value('white'), alt.value('black'))
            )
            return (heatmap + text).properties(title=title, width=280, height=220)
            
        with cm_col1:
            if "confusion_matrix" in lr_m:
                st.altair_chart(plot_cm(lr_m["confusion_matrix"], "Logistic Regression"), use_container_width=True)
        with cm_col2:
            if "confusion_matrix" in nb_m:
                st.altair_chart(plot_cm(nb_m["confusion_matrix"], "Multinomial Naive Bayes"), use_container_width=True)
                
        st.divider()
        
        # Top Informative Features
        st.subheader("Top Informative Features / Từ khóa Đặc trưng Nhất")
        st.markdown("Features with the highest mathematical influence on positive and negative predictions:")
        
        if top_features_dict and "Logistic Regression" in top_features_dict:
            lr_feats = top_features_dict["Logistic Regression"]
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                st.markdown("#### 🌟 Top Positive Indicators (Hệ số Dương cao nhất)")
                pos_df = pd.DataFrame(lr_feats.get("top_positive", [])[:15])
                chart_pos = alt.Chart(pos_df).mark_bar(color='#10B981').encode(
                    x=alt.X('weight:Q', title='Coefficient Weight'),
                    y=alt.Y('feature:N', sort='-x', title=None),
                    tooltip=['feature', 'weight']
                ).properties(height=360)
                st.altair_chart(chart_pos, use_container_width=True)
                
            with col_f2:
                st.markdown("#### 💣 Top Negative Indicators (Hệ số Âm cao nhất)")
                neg_df = pd.DataFrame(lr_feats.get("top_negative", [])[:15])
                chart_neg = alt.Chart(neg_df).mark_bar(color='#EF4444').encode(
                    x=alt.X('weight:Q', title='Negative Coefficient Weight'),
                    y=alt.Y('feature:N', sort='x', title=None),
                    tooltip=['feature', 'weight']
                ).properties(height=360)
                st.altair_chart(chart_neg, use_container_width=True)
    else:
        st.info("Metrics not loaded yet. Run training to generate evaluation charts.")

# ==========================================
# TAB 3: NLP MATH & FOUNDATIONS
# ==========================================
with tab3:
    st.subheader("NLP & Machine Learning Mathematical Bridge")
    st.markdown(
        "How unstructured human sentences are converted into computable matrix algebra for an introductory AI course."
    )
    
    st.markdown("### 1. The Vector Space Model (VSM) & Sparsity")
    st.write(
        "In traditional Machine Learning, algorithms cannot process raw strings. "
        "A text corpus of $N$ documents and a vocabulary of $V$ terms is represented as an $N \\times V$ matrix $\\mathbf{X}$:"
    )
    st.latex(r"\mathbf{X} = \begin{bmatrix} x_{1,1} & x_{1,2} & \cdots & x_{1,V} \\ x_{2,1} & x_{2,2} & \cdots & x_{2,V} \\ \vdots & \vdots & \ddots & \vdots \\ x_{N,1} & x_{N,2} & \cdots & x_{N,V} \end{bmatrix}")
    st.info(
        "💡 **Sparsity Concept**: Since a single movie review contains ~100 unique words out of a 10,000-word vocabulary, "
        "over **99% of entries are zero**. Scikit-learn stores this efficiently using Compressed Sparse Row (CSR) matrices."
    )
    
    st.markdown("### 2. TF-IDF Formula Breakdown")
    st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)")
    st.write(
        "- **Term Frequency $\\text{TF}(t, d)$**: How frequently term $t$ appears in document $d$ (often sublinearly scaled $1 + \\log(\\text{TF})$)."
    )
    st.write(
        "- **Inverse Document Frequency $\\text{IDF}(t, D)$**: Penalizes ubiquitous words (like *movie*, *film*) and elevates discriminative words (*masterpiece*, *atrocious*):"
    )
    st.latex(r"\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1")
    st.write("- **L2 Normalization**: Vectors are normalized to unit Euclidean length $\|\mathbf{x}\|_2 = 1$ to neutralize document length bias.")
    
    st.markdown("### 3. Classification Algorithms")
    c_algo1, c_algo2 = st.columns(2)
    with c_algo1:
        st.markdown("#### Logistic Regression (Discriminative)")
        st.write("Directly models the posterior probability $P(Y=1|X)$ using a linear hyper-plane mapped through a Sigmoid activation:")
        st.latex(r"P(Y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}")
        st.write("Optimized using Cross-Entropy Loss with L2 regularization:")
        st.latex(r"\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|^2")
        
    with c_algo2:
        st.markdown("#### Multinomial Naive Bayes (Generative)")
        st.write("Applies Bayes' Theorem assuming conditional independence between features given the class label:")
        st.latex(r"P(C_k|\mathbf{x}) = \frac{P(C_k) \prod_{i=1}^V P(w_i|C_k)^{x_i}}{P(\mathbf{x})}")
        st.write("With Laplace Smoothing to prevent zero probabilities for unseen test terms:")
        st.latex(r"\hat{P}(w_i|C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha V}")

# ==========================================
# TAB 4: ADVANCED EXTENSION (LLM ZERO-SHOT)
# ==========================================
with tab4:
    st.subheader("Advanced Extension: Traditional ML vs LLM Orchestration")
    st.markdown(
        "As an extension in modern AI curricula, compare how prompt-engineered Large Language Models (LLMs) "
        "perform zero-shot inference without prior domain-specific training compared to the trained Logistic Regression pipeline."
    )
    
    st.markdown("#### 🥊 Architectural Trade-off Matrix")
    matrix_df = pd.DataFrame(get_ml_vs_llm_comparison_matrix())
    st.dataframe(matrix_df, use_container_width=True)
    
    st.divider()
    st.markdown("#### 🧪 Test Zero-Shot LLM Reasoning on Current Review")
    
    llm_test_input = st.text_area(
        "Review to Evaluate with LLM:",
        value=user_review,
        height=90,
        key="llm_input"
    )
    
    api_key_input = st.text_input(
        "Optional: Gemini or OpenAI API Key (leave blank to run in simulated reasoning mode):",
        type="password"
    )
    
    if st.button("🤖 Run LLM Zero-Shot Analysis", type="secondary"):
        with st.spinner("Analyzing context, subtle tone, and narrative signals with LLM..."):
            llm_res = analyze_with_llm(llm_test_input, api_key=api_key_input)
            
            c_llm1, c_llm2 = st.columns([1.2, 2])
            with c_llm1:
                st.markdown(f"**Method**: `{llm_res.get('method')}`")
                sentiment_str = llm_res.get("sentiment", "positive").upper()
                conf = llm_res.get("confidence", 0.8)
                if sentiment_str == "POSITIVE":
                    st.markdown(f'<div class="badge-pos">👍 {sentiment_str} ({conf*100:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-neg">👎 {sentiment_str} ({conf*100:.1f}%)</div>', unsafe_allow_html=True)
                st.write("")
                st.write("**Key Signals Extracted:**")
                st.write(llm_res.get("key_signals", []))
                
            with c_llm2:
                st.markdown("**LLM Reasoning & Context Analysis:**")
                st.info(llm_res.get("reasoning", "No explanation provided."))
                
            with st.expander("🔍 View Prompt Template Orchestration"):
                st.code(llm_res.get("prompt_used", "N/A"), language="markdown")
