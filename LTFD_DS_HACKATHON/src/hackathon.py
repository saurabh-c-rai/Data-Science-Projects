# %%
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv(r"input\\train.csv", parse_dates=["application_date"])

# %%
df.head()

# %%
df["application_date"].max()

# %%
df.dtypes
# %%
sns.barplot(
    x=df["application_date"].dt.month, y=df["case_count"], hue="segment", data=df
)

# %%
print(np.dot([7, -4, -1], [8, -5, -6]))

# %%
