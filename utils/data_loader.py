import pandas as pd
import streamlit as st
import os


@st.cache_data
def load_dataset():

    dataset_path = "data/LSWMD_sample.pkl"

    # If dataset is available (local machine)
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

    # Streamlit Cloud (dataset not present)
    else:

        st.warning("Training dataset not available on Streamlit Cloud.")

        return None, None