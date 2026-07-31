import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_dataset():

    dataset_path = "data/LSWMD_sample.pkl"

    st.write("Current working directory:", os.getcwd())
    st.write("Dataset path:", dataset_path)
    st.write("Dataset exists:", os.path.exists(dataset_path))

    if os.path.exists(dataset_path):

        df = pd.read_pickle(dataset_path)

        df_clean = df[
            df["failureType"].apply(
                lambda x: len(x) > 0 and x[0][0] != "none"
            )
        ].copy()

        df_clean["failureLabel"] = df_clean["failureType"].apply(
            lambda x: x[0][0]
        )

        return df, df_clean

    st.warning("Training dataset not available on Streamlit Cloud.")
    return None, None