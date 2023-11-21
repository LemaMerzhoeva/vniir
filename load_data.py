import pandas as pd
from sqlalchemy import create_engine


def load_data():
    db_params = {'dbname': 'StreaX_db_vniir', 'user': 'postgres', 'password': '011009650702', 'host': 'gen-pvt'}

    time_intervals_file = 'time_stamps.csv'
    time_intervals_df = pd.read_csv(time_intervals_file)

    table_names = ['meas_data', 'calc_data', 'post_calc_data']

    engine = create_engine(
        f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["dbname"]}')

    for index, row in time_intervals_df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']

        excel_file_path = f'data_{index + 1}.xlsx'

        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as excel_writer:
            for table_name in table_names:
                # Формируем SQL-запрос
                sql_query = f"""
                    SELECT * FROM public.{table_name}
                    WHERE time_stamp BETWEEN timestamp '{start_time}' AND timestamp '{end_time}'
                    ORDER BY time_stamp
                """

                df = pd.read_sql_query(sql_query, engine)

                sheet_name = f'{table_name}'
                df.to_excel(excel_writer, index=False, sheet_name=sheet_name)

        print(f'Результаты сохранены в файл: {excel_file_path}')


