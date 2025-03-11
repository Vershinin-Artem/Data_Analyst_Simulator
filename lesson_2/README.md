# Задание 1

В наших данных использования ленты новостей есть два типа юзеров: те, кто пришел через платный трафик `source = 'ads'`, и те, кто пришел через органические каналы `source = 'organic'`.

Ваша задача — проанализировать и сравнить Retention этих двух групп пользователей. Решением этой задачи будет ответ на вопрос: **отличается ли характер использования приложения у этих групп пользователей?**

## Вопрос №1

Начнём с общего вопроса про Retention. Какое из утверждений об этой метрике справедливо?

- Единственный способ визуализации Retention — это тепловая карта
- Retention отражает долю пользователей, которые зарегистрировались в нашем приложении, от всех пользователей, которые его установили
- **Retention считается для конкретной когорты — пользователей, зарегистрировавшихся в приложении в определённый день**
- Абсолютное число пользователей в "нулевой" день не может быть больше 1

## Вопрос №2

Посмотрите на Retention для юзеров, пришедших через платный трафик. Каков он?

**Ответ:** `8.0`

## Вопрос №3

То же самое сделаем для "органических" пользователей. Какой Retention у них?

**Ответ:** `12.3`

## Вопрос №4

Какой вывод можно сделать? Как отличается характер использования приложения у этих групп пользователей?

**Выберите правильный ответ:**

На первый день пользователей с платного трафика обычно **больше**, чем с органического. В долгосрочной перспективе доля удержанных платных пользователей **меньше**, чем органических.

---

# Задание 2

За последние несколько дней у нас произошло два значительных события:

1. Наши маркетологи провели крупную рекламную кампанию, и в приложение пришло много новых пользователей!
2. В один из дней у нас резко упала аудитория.

Соответственно, перед нами стоят две задачи:

## 1. Проанализировать характер Retention пользователей, привлечённых рекламной кампанией

16 января наши маркетологи провели крупную рекламную кампанию, в результате чего количество новых пользователей ленты новостей в этот день составило **2592**. В то же время, благодаря органическому трафику, мы получили **741** нового пользователя.

Посмотрим Retention rate 1, 7 и 14 дня для когорты пользователей, привлечённых 16 января.

### SQL-запрос для определения Retention Rate:

```sql
WITH
t1 AS (SELECT COUNT(DISTINCT user_id) AS count_user_0_day
       FROM (SELECT user_id, MIN(toDate(time)) AS first_date
             FROM simulator_20250120.feed_actions
             WHERE source = 'ads'
             GROUP BY user_id
             HAVING first_date = toDate('2025-01-16'))),
             
t2 AS (SELECT COUNT(DISTINCT user_id) AS count_user_1_day
       FROM simulator_20250120.feed_actions
       WHERE source = 'ads'
       AND toDate(time) = toDate('2025-01-16') + 1
       AND user_id IN (SELECT user_id
                       FROM (SELECT user_id, MIN(toDate(time)) AS first_date
                             FROM simulator_20250120.feed_actions
                             WHERE source = 'ads'
                             GROUP BY user_id
                             HAVING first_date = toDate('2025-01-16')))),
                              
t3 AS (SELECT COUNT(DISTINCT user_id) AS count_user_7_day
       FROM simulator_20250120.feed_actions
       WHERE source = 'ads'
       AND toDate(time) = toDate('2025-01-16') + 7
       AND user_id IN (SELECT user_id
                       FROM (SELECT user_id, MIN(toDate(time)) AS first_date
                             FROM simulator_20250120.feed_actions
                             WHERE source = 'ads'
                             GROUP BY user_id
                             HAVING first_date = toDate('2025-01-16')))),
                              
t4 AS (SELECT COUNT(DISTINCT user_id) AS count_user_14_day
       FROM simulator_20250120.feed_actions
       WHERE source = 'ads'
       AND toDate(time) = toDate('2025-01-16') + 14
       AND user_id IN (SELECT user_id
                       FROM (SELECT user_id, MIN(toDate(time)) AS first_date
                             FROM simulator_20250120.feed_actions
                             WHERE source = 'ads'
                             GROUP BY user_id
                             HAVING first_date = toDate('2025-01-16'))))
                              
SELECT (SELECT count_user_0_day FROM t1) as count_user_0_day,
       (SELECT count_user_1_day FROM t2) as count_user_1_day,
       (SELECT count_user_7_day FROM t3) as count_user_7_day,
       (SELECT count_user_14_day FROM t4) as count_user_14_day,
       ((SELECT count_user_1_day FROM t2) /(SELECT count_user_0_day FROM t1))*100 as RR_1_day,
       ((SELECT count_user_7_day FROM t3) /(SELECT count_user_0_day FROM t1))*100 as RR_7_day,
       ((SELECT count_user_14_day FROM t4) /(SELECT count_user_0_day FROM t1))*100 as RR_14_day;
```

### Retention Rate для платного трафика:
- **1-й день:** `3.97%`
- **7-й день:** `2.89%`
- **14-й день:** `1.89%`

### Retention Rate для органического трафика:
- **1-й день:** `29.96%`
- **7-й день:** `26.32%`
- **14-й день:** `14.84%`

Исходя из этих данных, можно предположить, что у нас в сервисе низкий процент удержания пользователей, привлечённых через рекламу. В то же время органический трафик демонстрирует высокий уровень удержания. Это может свидетельствовать о нерелевантности привлекаемой рекламной аудитории.

## 2. Выяснить, какие пользователи не смогли воспользоваться лентой. Что их объединяет?

25 января 2025 года наблюдалось **значительное снижение активности пользователей ленты новостей на 13%**. Для выявления причин падения посещаемости проведён анализ данных в разрезе городов.

### SQL-запрос для анализа посещаемости по городам:

```sql
WITH
t1 AS (SELECT COUNT(DISTINCT user_id) AS users, city
       FROM simulator_20250120.feed_actions
       WHERE toDate(time) = '2025-01-24'
       GROUP BY city),

t2 AS (SELECT COUNT(DISTINCT user_id) AS users, city
       FROM simulator_20250120.feed_actions
       WHERE toDate(time) = '2025-01-25'
       GROUP BY city)

SELECT t1.users AS date_01_24,
       t2.users AS date_01_25,
       city
FROM t1
LEFT JOIN t2 USING(city)
LIMIT 5;
```

**Результаты анализа показали, что пользователи из топ-4 городов практически не проявляли активности 25 января, что указывает на возможные технические проблемы.**


