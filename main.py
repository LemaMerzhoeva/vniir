from correllation import calculate_and_save_averages
from file_parse import process_files_parallel
from load_data import load_data

if __name__ == '__main__':
    print("Старт работы ...")
    # load_data()
    # calculate_and_save_averages('data', 'general_data.xlsx')

    process_files_parallel('data')
    print("Работа выполнена")

