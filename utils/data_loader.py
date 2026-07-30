import pandas as pd
import streamlit as st


@st.cache_data
def load_dataset():

    df = pd.read_pickle("data/LSWMD.pkl")

    df_clean = df[
        df["failureType"].apply(
            lambda x: len(x) > 0 and x[0][0] != "none"
        )
    ].copy()

    df_clean["failureLabel"] = df_clean["failureType"].apply(
        lambda x: x[0][0]
    )

    return df, df_clean