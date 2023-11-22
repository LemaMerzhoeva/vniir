from correllation import calculate_and_save_averages
from file_parse import process_files_parallel_histogram, process_files_parallel_boxplot
from load_data import load_data

if __name__ == '__main__':
    print("Старт работы ...")
    # load_data()
    # calculate_and_save_averages('data', 'general_data.xlsx')

    # process_files_parallel_histogram('data')
    process_files_parallel_boxplot('data/general_data')
    print("Работа выполнена")

