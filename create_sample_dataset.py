import pandas as pd

# Load original dataset
df = pd.read_pickle("data/LSWMD.pkl")

# Keep only wafers with defects
df_clean = df[
    df["failureType"].apply(
        lambda x: len(x) > 0 and x[0][0] != "none"
    )
]

# Take first 200 samples
sample_df = df_clean.sample(
    n=200,
    random_state=42
)

# Save
sample_df.to_pickle("data/LSWMD_sample.pkl")

print("✅ Sample dataset created!")
print(sample_df.shape)