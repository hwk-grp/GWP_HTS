import pandas
from rdkit import Chem
from rdkit.Chem import PandasTools

dataset_name = 'initial_dataset'

# Load the dataset
df = pandas.read_excel(f'./dataset/{dataset_name}.xlsx')


# Define the SMARTS patterns for the functional groups you want to remove
smarts_patterns = {
    "nitro group": "[N+](=O)[O-]",  # Nitro
    "peroxide": "[O]~[O]",  # Peroxide
    "3-member ring": "[r3]",  # Any 3-member ring
    "4-member ring": "[r4]",  # Any 4-member ring
    "disulfide": "S-S",  # Disulfide
    "C=S group": "[C]=S",  # Thiocarbonyl (C=S)
    "linear diene": "[C]=C-[C]=C",  # Conjugated diene
    "NO group": "N=O",  # NO group
    "NN group": "N~N",  # NN group (any bond order)
    "N-X group": "[N]-[F,Cl,Br]",  # N-F, N-Cl, or N-Br
    "O-X group": "[O]-[F,Cl,Br]",  # O-F, O-Cl, or O-Br
    "C=C=C group": "C=C=C",  # Allene (C=C=C group)
    "keto–enol tautomerism": "[#6][C](=O)[C](O)[#6]",  # Keto-enol tautomerism
    "OF group": "[O]-[F]",  # OF group (oxygen-fluorine bond)
    "=CF2 group": "[C](=F)(-F)"  # CF2 group (carbon with two fluorines and double bond)
}

# Function to check if a molecule contains any of the specified functional groups
def contains_functional_group(mol, patterns):
    if '#' in mol:
        return 1
    mol = Chem.MolFromSmiles(mol)
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            return 1
    return 0

df['contains_excluded_groups'] = df['Key'].apply(lambda mol: contains_functional_group(mol, smarts_patterns))

# Save the original DataFrame with the new column
df.to_excel('./preds/substructure_result.xlsx', index=False)

'''
# Create a filtered DataFrame excluding molecules that contain the unwanted functional groups
df_filtered = df[df['contains_excluded_groups'] == False]

# Save the filtered DataFrame
df_filtered.to_excel('filtered_smiles_dataset.csv', index=False)
'''
