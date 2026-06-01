import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="AIMS DTU Evaluation Dashboard", layout="wide")

st.title("Comparative Analysis of Latent Space Geometries (SimCLR vs MAE)")
st.caption("AIMS DTU Research Internship Assignment 2026 Submission")

# Try loading t-SNE evaluation vectors
try:
    data = np.load('tsne_results.npz')
    sim_tsne = data['sim_tsne']
    mae_tsne = data['mae_tsne']
    labels = data['labels'].astype(str)
    
    df_sim = pd.DataFrame({'X': sim_tsne[:, 0], 'Y': sim_tsne[:, 1], 'Class': labels})
    df_mae = pd.DataFrame({'X': mae_tsne[:, 0], 'Y': mae_tsne[:, 1], 'Class': labels})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SimCLR Clustering Properties (Contrastive Space)")
        fig_sim = px.scatter(df_sim, x='X', y='Y', color='Class', title="SimCLR Space Geometry")
        st.plotly_chart(fig_sim, use_container_width=True)
        st.info("💡 Discriminative objectives force sharp spatial boundary separation.")

    with col2:
        st.subheader("MAE Clustering Properties (Reconstructive Space)")
        fig_mae = px.scatter(df_mae, x='X', y='Y', color='Class', title="MAE Space Geometry")
        st.plotly_chart(fig_mae, use_container_width=True)
        st.info("💡 Generative frameworks preserve global context but display overlapping structures.")

except FileNotFoundError:
    st.error("Please run `train.py` and `eval.py` sequentially to populate presentation arrays.")