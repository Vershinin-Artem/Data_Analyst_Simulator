# ETL Pipeline для расчёта метрик в Airflow

Создадим ETL пайплайн, итогом которого будет DAG в Airflow, считающий метрики каждый день за вчера.

## Этапы пайплайна

1. **Обработка исходных таблиц**  
   - Таблица `feed_actions`: для каждого пользователя считаем:
     - Число просмотров контента
     - Число лайков контента
   - Таблица `message_actions`: для каждого пользователя считаем:
     - Сколько сообщений получил и отправил
     - Скольким людям написал
     - Сколько людей ему написали  
   *Каждая выгрузка должна быть в отдельном таске.*

2. **Объединение данных**  
   Объединяем две таблицы в одну.

3. **Расчёт метрик по срезам**  
   Для объединённой таблицы считаем метрики в разрезе:
   - По полу (`gender`)
   - По возрасту (`age`)
   - По ОС (`os`)  
   *Делаем три разных таска на каждый срез.*

4. **Запись результатов**  
   Финальные данные со всеми метриками записываем в отдельную таблицу в ClickHouse.

5. **Ежедневное обновление**  
   Каждый день таблица должна дополняться новыми данными.

## Структура финальной таблицы

| Поле | Название в таблице | Описание |
|------|--------------------|----------|
| Дата | `event_date` | Дата расчёта метрик |
| Название среза | `dimension` | Тип среза (`os`/`gender`/`age`) |
| Значение среза | `dimension_value` | Конкретное значение среза |
| Число просмотров | `views` | Количество просмотров контента |
| Число лайков | `likes` | Количество лайков контента |
| Число полученных сообщений | `messages_received` | Количество входящих сообщений |
| Число отправленных сообщений | `messages_sent` | Количество исходящих сообщений |
| От скольких пользователей получили сообщения | `users_received` | Количество уникальных отправителей |
| Скольким пользователям отправили сообщение | `users_sent` | Количество уникальных получателей |

**Примечание:** Срезы — это `os`, `gender` и `age`.

## Требования к реализации
- Таблица должна быть загружена в схему `test`
- Ответ на задание — название таблицы в схеме `test`

# ETL Pipeline в Airflow для ежедневных метрик

Полный код DAG для расчета ежедневных метрик:

```python
from datetime import datetime, timedelta  # Для работы с датами
import pandas as pd
import pandahouse as ph
from airflow.decorators import dag, task


# Конфигурация подключений к ClickHouse


# Подключение к исходной БД (simulator)
connection = {
    'host': 'https://clickhouse.lab.karpov.courses',
    'database': 'simulator_20250120',
    'user': 'student',
    'password': 'dpo_python_2020'
}

# Подключение к тестовой БД для записи результатов
connection_test = {
    'host': 'https://clickhouse.lab.karpov.courses',
    'database': 'test',
    'user': 'student-rw',
    'password': '656e2b0c9c'
}


# Настройки DAG


default_args = {
    'owner': 'A_Vershinin',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 19),
}

schedule_interval = '0 15 * * *'  # Запуск ежедневно в 15:00

# Основные функции ETL-процесса


@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def daily_uploads_VA():
    
    @task()
    def extract_feed_actions():
        """Извлекает данные о просмотрах и лайках из feed_actions"""
        q = '''
        SELECT 
            user_id,
            toDate(time) AS event_date,
            SUM(action = 'view') AS views,
            SUM(action = 'like') AS likes
        FROM simulator_20250120.feed_actions
        WHERE event_date = today() - 1
        GROUP BY user_id, event_date
        '''
        return ph.read_clickhouse(q, connection=connection)

    @task()
    def extract_message_actions():
        """Извлекает данные о сообщениях из message_actions"""
        q = '''
        WITH
        t1 AS (
            SELECT 
                receiver_id,
                toDate(time) AS event_date,
                COUNT(receiver_id) AS messages_received,
                COUNT(DISTINCT user_id) AS users_received
            FROM simulator_20250120.message_actions
            WHERE event_date = today() - 1
            GROUP BY receiver_id, event_date
        ),
        t2 AS (
            SELECT 
                user_id,
                toDate(time) AS event_date,
                COUNT(user_id) AS messages_sent,
                COUNT(DISTINCT receiver_id) AS users_sent
            FROM simulator_20250120.message_actions
            WHERE event_date = today() - 1
            GROUP BY user_id, event_date, age, gender, os
        )
        SELECT 
            COALESCE(user_id, receiver_id) as user_id,
            (today() - 1) AS event_date,
            messages_received,
            messages_sent,
            users_received,
            users_sent
        FROM t1
        FULL OUTER JOIN t2 ON t1.receiver_id = t2.user_id
        '''
        return ph.read_clickhouse(q, connection=connection)
    
    @task()
    def extracts_date(df_feed_actions, df_message_actions):
        """Объединяет данные и добавляет информацию о срезах (os, gender, age)"""
        # Получаем данные о срезах из обеих таблиц
        q_feed = "SELECT DISTINCT user_id, os, gender, age FROM simulator_20250120.feed_actions"
        q_message = "SELECT DISTINCT user_id, os, gender, age FROM simulator_20250120.message_actions"
        
        df_feed_dim = ph.read_clickhouse(q_feed, connection=connection)
        df_message_dim = ph.read_clickhouse(q_message, connection=connection)
        
        # Объединяем и удаляем дубликаты
        df_dimension = pd.concat([df_feed_dim, df_message_dim]).drop_duplicates('user_id')
        
        # Объединяем все данные
        df_merged = (
            df_feed_actions.merge(df_message_actions, on=['event_date', 'user_id'], how='outer')
                          .merge(df_dimension, on='user_id', how='left')
        )
        return df_merged
        
    @task()
    def slice_os(df_merge):
        """Считает метрики по срезу операционных систем"""
        df_os = (df_merge.groupby(['event_date', 'os'], as_index=False)
                         .sum()
                         .rename(columns={'os': 'dimension_value'}))
        df_os.insert(1, 'dimension', 'os')
        return df_os

    @task()
    def slice_gender(df_merge):
        """Считает метрики по срезу гендера"""
        df_gender = (df_merge.groupby(['event_date', 'gender'], as_index=False)
                             .sum()
                             .rename(columns={'gender': 'dimension_value'}))
        df_gender.insert(1, 'dimension', 'gender')
        return df_gender

    @task()
    def slice_age(df_merge):
        """Считает метрики по срезу возраста"""
        df_age = (df_merge.groupby(['event_date', 'age'], as_index=False)
                          .sum()
                          .rename(columns={'age': 'dimension_value'}))
        df_age.insert(1, 'dimension', 'age')
        return df_age
        
    @task()
    def load(df_os, df_gender, df_age):
        """Загружает финальные данные в ClickHouse"""
        # Объединяем все срезы
        df_final = pd.concat([df_os, df_gender, df_age])
        
        # Приводим типы данных
        df_final = df_final.astype({
            'views': 'int',
            'likes': 'int',
            'messages_received': 'int',
            'messages_sent': 'int',
            'users_received': 'int',
            'users_sent': 'int'
        })
        
        # Создаем таблицу (если не существует)
        create_table = """
        CREATE TABLE IF NOT EXISTS test.A_Vershinin (
            event_date Date,
            dimension String,
            dimension_value String,
            likes Int64,
            views Int64,
            messages_received Int64,
            messages_sent Int64,
            users_received Int64,
            users_sent Int64
        ) ENGINE = MergeTree()
        ORDER BY event_date
        """
        ph.execute(create_table, connection=connection_test)
        
        # Загружаем данные
        ph.to_clickhouse(df=df_final, table='A_Vershinin', 
                        index=False, connection=connection_test)

    # Оркестрация задач
  
    df_feed = extract_feed_actions()
    df_message = extract_message_actions()
    df_merged = extracts_date(df_feed, df_message)
    
    df_os = slice_os(df_merged)
    df_gender = slice_gender(df_merged)
    df_age = slice_age(df_merged)
    
    load(df_os, df_gender, df_age)

# Инициализация DAG
v5_daily_uploads_VA = daily_uploads_VA()
