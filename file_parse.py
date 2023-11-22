import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

def process_file_histogram(file):
    try:
        print([file])
        folder_path = 'data/general_data'
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name=None)

        output_folder = os.path.join(folder_path, f"{file}_histograms")
        os.makedirs(output_folder, exist_ok=True)

        for sheet_name, sheet_data in df.items():
            if not sheet_data.empty:
                numeric_columns = [col for col in sheet_data.columns if pd.api.types.is_numeric_dtype(sheet_data[col])]

                if numeric_columns:
                    for col in numeric_columns:
                        plt.figure()

                        sheet_data[col].hist(alpha=0.5, bins=20)
                        plt.title(f"{sheet_name} - {col}")
                        plt.xlabel("Values")
                        plt.ylabel("Frequency")

                        plot_filename = os.path.join(output_folder, f"{sheet_name}_{col}_histogram.jpg")
                        plt.savefig(plot_filename, format='jpg')
                        plt.close()
    except Exception as e:
        print(e)

def process_files_parallel_histogram(folder_path):
    files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

    with Pool() as pool:
        pool.map(process_file_histogram, files)

def process_file_boxplot(file):
    try:
        print([file])
        folder_path = 'data/general_data'
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name=None)

        # Create a subfolder for each file
        output_folder = os.path.join(folder_path, f"{file}_boxplots")
        os.makedirs(output_folder, exist_ok=True)

        for sheet_name, sheet_data in df.items():
            if not sheet_data.empty:
                # Filter numeric columns
                numeric_columns = [col for col in sheet_data.columns if pd.api.types.is_numeric_dtype(sheet_data[col])]

                if numeric_columns:
                    # Iterate over numeric columns and create a separate boxplot for each
                    for col in numeric_columns:
                        plt.figure()

                        sheet_data.boxplot(column=col)
                        plt.title(f"{sheet_name} - {col} Boxplot")
                        plt.xlabel("Column")
                        plt.ylabel("Values")

                        plot_filename = os.path.join(output_folder, f"{sheet_name}_{col}_boxplot.jpg")
                        plt.savefig(plot_filename, format='jpg')
                        plt.close()
    except Exception as e:
        print(e)

def process_files_parallel_boxplot(folder_path):
    files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

    with Pool() as pool:
        pool.map(process_file_boxplot, files)
