import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

def process_file(file):
    try:
        print([file])
        folder_path = 'data'
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name=None)

        # Create a subfolder for each file
        output_folder = os.path.join(folder_path, f"{file}_histograms")
        os.makedirs(output_folder, exist_ok=True)

        for sheet_name, sheet_data in df.items():
            if not sheet_data.empty:
                # Filter numeric columns
                numeric_columns = [col for col in sheet_data.columns if pd.api.types.is_numeric_dtype(sheet_data[col])]

                if numeric_columns:
                    # Iterate over numeric columns and create a separate histogram for each
                    for col in numeric_columns:
                        plt.figure()

                        sheet_data[col].hist(alpha=0.5, bins=20)
                        plt.title(f"{sheet_name} - {col}")
                        plt.xlabel("Values")
                        plt.ylabel("Frequency")

                        # Save the plot as a JPG image in the subfolder
                        plot_filename = os.path.join(output_folder, f"{sheet_name}_{col}_histogram.jpg")
                        plt.savefig(plot_filename, format='jpg')
                        plt.close()  # Close the current figure
    except Exception as e:
        print(e)

def process_files_parallel(folder_path):
    files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

    # Use multiprocessing to parallelize the processing of files
    with Pool() as pool:
        pool.map(process_file, files)