import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to molecule
    return Chem.MolToSmiles(mol, canonical=True) if mol else None  # Canonicalize

df_A = pd.read_excel('../dataaset/initial_dataset.xlsx')
df_B = pd.read_excel('pu_result.xlsx')


# Ensure both datasets have the same column names for the first column
df_A.rename(columns={df_A.columns[0]: 'Key'}, inplace=True)
df_B.rename(columns={df_B.columns[0]: 'Key'}, inplace=True)

df_B['Key'] = df_B['Key'].apply(canonicalize_smiles)



# Merge B's rows into A based on the 'Key' column
# This performs a left merge, keeping all of A's rows and matching B's rows
merged_df = pd.merge(df_A, df_B, on='Key', how='left')

# Save the result to a new Excel file
merged_df.to_excel('pu_cid_ascending_result.xlsx', index=False)

print("Merged file saved as 'merged_A_B.xlsx'")

