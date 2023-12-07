import correllation
import file_parse
import load_data
import regres

if __name__ == '__main__':
    print("Старт работы ...")
    # load_data()
    # calculate_and_save_averages('data/general_data', 'general_data.xlsx')

    # process_files_parallel_histogram('data/general_data')
    # process_files_parallel_boxplot('data/general_data')

    # correllation.corellation_matrix('general_data.xlsx')
    regres.regres_linear_analysis('general_data.xlsx')
    regres.non_linear_regression_analysis('general_data.xlsx', 'linear')
    # regres.non_linear_regression_analysis('general_data.xlsx', 'exp')
    regres.non_linear_regression_analysis('general_data.xlsx', 'quadratic')
    # regres.non_linear_regression_analysis('general_data.xlsx', 'log')
    print("Работа выполнена")

