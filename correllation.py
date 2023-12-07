import os

def calculate_and_save_averages(folder_path, output_file):

    files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]


    result_dfs = {"meas_data": pd.DataFrame(columns=["Имя файла"]),
                  "calc_data": pd.DataFrame(columns=["Имя файла"]),
                  "post_calc_data": pd.DataFrame(columns=["Имя файла"])}

    for file in files:
        print([file])

        df_meas = pd.read_excel(os.path.join(folder_path, file), sheet_name='meas_data')
        df_calc = pd.read_excel(os.path.join(folder_path, file), sheet_name='calc_data')
        df_post_calc = pd.read_excel(os.path.join(folder_path, file), sheet_name='post_calc_data')


        if not df_meas.empty:

            averages_meas = df_meas.apply(lambda col: col.mean() if pd.api.types.is_numeric_dtype(col) else 'text')


            result_dfs["meas_data"] = pd.concat([result_dfs["meas_data"], pd.DataFrame({"Имя файла": [file], **averages_meas})], ignore_index=True)

        if not df_calc.empty:
            averages_calc = df_calc.apply(lambda col: col.mean() if pd.api.types.is_numeric_dtype(col) else 'text')
            result_dfs["calc_data"] = pd.concat([result_dfs["calc_data"], pd.DataFrame({"Имя файла": [file], **averages_calc})], ignore_index=True)

        if not df_post_calc.empty:
            averages_post_calc = df_post_calc.apply(lambda col: col.mean() if pd.api.types.is_numeric_dtype(col) else 'text')
            result_dfs["post_calc_data"] = pd.concat([result_dfs["post_calc_data"], pd.DataFrame({"Имя файла": [file], **averages_post_calc})], ignore_index=True)


    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, result_df in result_dfs.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def corellation_matrix(input_file_path):
    df1 = pd.read_excel(input_file_path, sheet_name='meas_data', engine='openpyxl')
    df2 = pd.read_excel(input_file_path, sheet_name='calc_data', engine='openpyxl')

    merged_df = pd.merge(df1, df2, on='Имя файла')

    numeric_columns = merged_df.select_dtypes(include='number').columns
    cleaned_df = merged_df[numeric_columns].dropna(axis=1)

    columns_to_remove = [col for col in cleaned_df.columns if '_id' in col.lower()]
    cleaned_df = cleaned_df.drop(columns=columns_to_remove)

    correlation_matrix_pearson = cleaned_df.corr(method='pearson')
    correlation_matrix_spearman = cleaned_df.corr(method='spearman')
    correlation_matrix_kendall = cleaned_df.corr(method='kendall')

    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set(font_scale=0.5)

    correlation_matrix_pearson_rounded = np.ceil(correlation_matrix_pearson * 100) / 100
    correlation_matrix_spearman_rounded = np.ceil(correlation_matrix_spearman * 100) / 100
    correlation_matrix_kendall_rounded = np.ceil(correlation_matrix_kendall * 100) / 100

    sns.heatmap(correlation_matrix_pearson_rounded, cmap='RdBu_r', annot=True, fmt=".2f",
                cbar_kws={'label': 'Correlation'}, vmax=1, vmin=-1)
    plt.title('Pearson Correlation')
    plt.savefig('pearson_correlation.png', dpi=300)
    plt.show()

    sns.heatmap(correlation_matrix_spearman_rounded, cmap='RdBu_r', annot=True, fmt=".2f",
                cbar_kws={'label': 'Correlation'}, vmax=1, vmin=-1)
    plt.title('Spearman Correlation')
    plt.savefig('spearman_correlation.png', dpi=300)
    plt.show()

    sns.heatmap(correlation_matrix_kendall_rounded, cmap='RdBu_r', annot=True, fmt=".2f",
                cbar_kws={'label': 'Correlation'}, vmax=1, vmin=-1)
    plt.title('Kendall Correlation')
    plt.savefig('kendall_correlation.png', dpi=300)
    plt.show()