import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
file_names = [
    'Onehot.csv',
    'DNA2vec.csv',
    'DNABERT.csv',
    'MGW.csv',
    'Shear.csv',
    'Stretch.csv',
    'Stagger.csv',
    'Buckle.csv',
    'ProT.csv',
    'Opening.csv',
    'Shift.csv',
    'Slide.csv',
    'Rise.csv',
    'Tilt.csv',
    'Roll.csv',
    'HelT.csv'
]

dataframes = []

for i, file_name in enumerate(file_names):
    df = pd.read_csv(file_name, header=None)  
    if 3 <= i < 16: 
        df = df.iloc[:, 2:]  
    dataframes.append(df)

combined_df = pd.concat(dataframes, axis=1)

feature_importance= pd.read_csv("Rearrange_101nt_feature_importance_sorted.csv")

selected_column_indexes = feature_importance.iloc[:100, 0].astype(int).values

if all(0 <= idx < combined_df.shape[1] for idx in selected_column_indexes):

    selected_data = combined_df.iloc[:, selected_column_indexes]


    scaler = StandardScaler()
    selected_data_standardized = scaler.fit_transform(selected_data)

    min_max_scaler = MinMaxScaler()
    selected_data_normalized = min_max_scaler.fit_transform(selected_data)

    new_data = pd.DataFrame(selected_data_normalized,
                            columns=[f"Column_{index}_normalized" for index in selected_column_indexes])

    new_csv_path = "Rearrange_101nt_features.csv"
    new_data.to_csv(new_csv_path, index=False)
else:
    print("Selected column indexes are out of range.")