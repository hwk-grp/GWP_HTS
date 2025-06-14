import pandas as pd

# Load the Excel file
file_path = "arranged_result.xlsx"
df = pd.read_excel(file_path)

# Apply filtering conditions
filtered_df = df[
    (df["GWP_log"] < 2) &
    (df["tau_ipcc"] < 0) &
    (df["mol_re"] < 0.15) &
    (df["Safety score"] > 0.5) &
    (df["LFL"] > 0.1) &
    (df["Tc"] < 500) &
    (df["Unstable"] == 0)
]

initial_count = df.shape[0]

count_gwp = df[df["GWP_log"] < 2].shape[0]
count_tau = df[df["tau_ipcc"] < 0].shape[0]
count_mol_re = df[df["mol_re"] < 0.15].shape[0]
count_Safety_score = df[df["Safety score"] > 0.5].shape[0]
count_LFL = df[df["LFL"] > 0.1].shape[0]
count_Tc = df[df["Tc"] < 500].shape[0]
count_Unstable = df[df["Unstable"] == 0].shape[0]

print(f"Number of data points with GWP_log: {count_gwp}")
print(initial_count-count_gwp)
print(f"Number of data points with tau_ipcc: {count_tau}")
print(initial_count-count_tau)
print(f"Number of data points with mol_re: {count_mol_re}")
print(initial_count-count_mol_re)
print(f"Number of data points with Safety score: {count_Safety_score}")
print(initial_count-count_Safety_score)
print(f"Number of data points with LFL: {count_LFL}")
print(initial_count-count_LFL)
print(f"Number of data points with Tc: {count_Tc}")
print(initial_count-count_Tc)
print(f"Number of data points with Unstable: {count_Unstable}")
print(initial_count-count_Unstable)

# filtered_df.to_excel("filtered_result.xlsx", index=False)

# Print the number of filtered data points
print(f"Number of filtered data points: {filtered_df.shape[0]}")
