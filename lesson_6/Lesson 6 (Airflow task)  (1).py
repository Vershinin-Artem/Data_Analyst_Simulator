
from datetime import datetime, timedelta #D
import pandas as pd
import pandahouse as ph
from io import StringIO
import requests

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

# Подключаемся к исходной БД
connection = {'host': 'https://clickhouse.lab.karpov.courses',
'database':'simulator_20250120',
'user':'student',
'password':'dpo_python_2020'}

# Подключаемся к тестовой БД
connection_test = {'host': 'https://clickhouse.lab.karpov.courses',
                      'database':'test',
                      'user':'student-rw', 
                      'password':'656e2b0c9c'}

# Дефолтные параметры, которые прокидываются в таски
default_args = {
    'owner': 'A_Vershinin',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 19),
}

# Интервал запуска DAG
schedule_interval = '0 15 * * *'

@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def daily_uploads_VA():
    
    @task()
    def extract_feed_actions():    # Выгрузим лайки и просмотры
        q_1 = ''' SELECT user_id,
                         toDate(time) AS event_date,
                         SUM(action = 'view') AS views,
                         SUM(action = 'like') AS likes
                  FROM simulator_20250120.feed_actions
                  WHERE event_date = today() - 1
                  GROUP BY user_id, event_date

             '''
        df_feed_actions = ph.read_clickhouse(q_1, connection=connection)
        return df_feed_actions


    @task()
    def extract_message_actions():  # Выгрузим информацию по полученным/ отправленным сообщениям
        q_2 = '''WITH
                -- количество полученных сообщений и количество получателей
                t1 AS (SELECT receiver_id,
                              toDate(time) AS event_date,
                              COUNT(receiver_id) AS messages_received,
                              COUNT(DISTINCT user_id) AS users_received
                       FROM simulator_20250120.message_actions
                       WHERE event_date = today() - 1
                       GROUP BY receiver_id, event_date),
                -- количество отправленных сообщений и количество отправителей
                t2 AS (SELECT user_id,
                              toDate(time) AS event_date,
                              COUNT(user_id) AS messages_sent,
                              COUNT(DISTINCT receiver_id) AS users_sent
                       FROM simulator_20250120.message_actions
                       WHERE event_date = today() - 1
                       GROUP BY user_id, event_date, age, gender, os)

                SELECT COALESCE(user_id, receiver_id) as user_id,
                       (today() - 1)  AS event_date,
                       messages_received,
                       messages_sent,
                       users_received,
                       users_sent
                FROM t1
                FULL OUTER JOIN t2
                ON t1.receiver_id = t2.user_id
                           '''
        df_message_actions = ph.read_clickhouse(q_2, connection=connection)
        return df_message_actions
    
    @task()
    def extracts_date(df_feed_actions, df_message_actions): # данные для срезов
        q_3 = """
            SELECT DISTINCT
                user_id,
                os,
                gender,
                age
            FROM simulator_20250120.feed_actions
            """

        q_4 = """
            SELECT DISTINCT
                user_id,
                os,
                gender,
                age
            FROM simulator_20250120.message_actions
            """
            
        df_feed_actions_dimension = ph.read_clickhouse(q_3, connection=connection)
        df_message_actions_dimension = ph.read_clickhouse(q_4, connection=connection)
        df_dimension = pd.concat([df_feed_actions_dimension, df_message_actions_dimension], ignore_index=True)
        df_dimension = df_dimension.drop_duplicates(subset=['user_id'])

        # Объединение ранне выгруженных данных
        df_feed_message =  df_feed_actions.merge(df_message_actions, on=['event_date', 'user_id'], how='outer')
        df_merge = df_feed_message.merge(df_dimension, on='user_id', how='left')
        return df_merge
        
        #срез по операционной системе устройства
    @task()
    def slice_os(df_merge):
        df_os = df_merge[['event_date','os','views','likes','messages_received','messages_sent','users_received',
        'users_sent']].groupby(['event_date', 'os'], as_index=False).sum().rename(columns={'os':'dimension_value'})
        dimension_column = 'os'
        df_os.insert(1, 'dimension', dimension_column)
        return df_os

        #срез по полу
    @task()
    def slice_gender(df_merge):
        df_gender = df_merge[['event_date','gender','views','likes','messages_received','messages_sent','users_received',                       'users_sent']].groupby(['event_date', 'gender'],as_index=False).sum().rename(columns={'gender':'dimension_value'})
        dimension_column = 'gender'
        df_gender.insert(1, 'dimension', dimension_column)
        return df_gender

        #срез по возрасту
    @task()
    def slice_age(df_merge):
        df_age = df_merge[['event_date','age','views','likes','messages_received','messages_sent','users_received',
        'users_sent']].groupby(['event_date', 'age'], as_index=False).sum().rename(columns={'age':'dimension_value'})
        dimension_column = 'age'
        df_age.insert(1, 'dimension', dimension_column)
        return df_age
        
    #выгрузка в тестовую таблицу
    @task()
    def load(df_os, df_gender, df_age):
        df_concat = pd.concat([df_os, df_gender, df_age])

        df_concat = df_concat.astype({
                    'views': 'int',
                    'likes': 'int',
                    'messages_received': 'int',
                    'messages_sent': 'int',
                    'users_received': 'int',
                    'users_sent': 'int'})

        # Создание таблицы
        final_table = """
                    CREATE TABLE IF NOT EXISTS test.A_Vershinin
                            (event_date Date,
                            dimension String,
                            dimension_value String,
                            likes Int64,
                            views Int64,    
                            messages_received Int64,     
                            messages_sent Int64,     
                            users_received Int64,    
                            users_sent Int64)
                            ENGINE = MergeTree()
                            ORDER BY event_date"""
        ph.execute(final_table, connection=connection_test)
        ph.to_clickhouse(df=df_concat, table='A_Vershinin', index=False, connection=connection_test)

    df_feed_actions = extract_feed_actions()
    df_message_actions = extract_message_actions()
    df_merge = extracts_date(df_feed_actions, df_message_actions)
    df_os = slice_os(df_merge)
    df_gender = slice_gender(df_merge)
    df_age = slice_age(df_merge)
    load(df_os, df_gender, df_age)

v5_daily_uploads_VA = daily_uploads_VA()

    