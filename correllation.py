import os
import pandas as pd

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
