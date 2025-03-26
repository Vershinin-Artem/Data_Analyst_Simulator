# Прогнозирование метрик

Поддерживать и стимулировать пользовательскую активность – важная задача для продукта, подобного нашему. Для этого наша команда маркетологов решила организовать флэшмоб в ленте новостей: участники должны сделать пост, где они рассказывают какой-то интересный факт о себе, и опубликовать его с хэштегом. Три поста, собравших наибольшее число лайков, получают призы.

Флэшмоб проходил с 2025-01-17 по 2025-01-23. Оценим эффективность этого мероприятия.

Выберите те варианты из списка, которые нам доступны и которые было бы осмысленно тестировать.

- Средний возраст пользователей, зарегистрировашихся до начала флэшмоба
- **CTR**
- **DAU**
- Заинтересованность наших пользователей в продукте
- **Число лайков на пользователя**
- Число комментариев под постами
- **Число просматриваемых постов**

Отлично! Настало время использовать CausalImpact, чтобы заключить о наличии либо отсутствии эффекта — а также насколько этот эффект выражен.

Первым делом надо определиться, какой диапазон дат мы возьмём в качестве пост-периода. Выберите наиболее удачный вариант.

- **Весь период флэшмоба**
- Неделя-две после окончания флэшмоба
- Первые день-два с начала флэшмоба
- Весь период флэшмоба + неделя-две после его окончания

Теперь определимся с пре-периодом, т.е. периодом до начала флэшмоба. Сколько будем брать? Выберите наиболее удачный вариант.

- Не менее 2-3 месяцев
- 1 неделя
- Более одного дня, но меньше недели
- **2 недели и более**

Успех CausalImpact зависит не только от размера данных, но и от правильного выбора модели. Идеальный случай — это когда у нас есть ковариаты. В данном случае подходящих вариантов нет, но если, скажем, флэшмоб видели только жители Москвы, то в качестве ковариаты могла бы выступить метрика из другого города (или нескольких).

Впрочем, некоторые фокусы с построением моделей относятся и к нашим данным. Например, если мы возьмём метрики с часовым разрешением, то стандартные настройки CausalImpact нам не подойдут. Почему?

- Единичный корень временного ряда перестанет сходиться
- Будут огромные доверительные интервалы, и ничего не будет значимо
- **У нас есть выраженная внутридневная сезонность, не включённая в модель**
- Почасовое разрешение увеличивает размер данных, и модель начинает долго считаться

Ещё одна вещь, которую можно учесть — это включить тренд в модель. Напрямую через аргументы модели это не делается, но можно построить модель прямо в TensorFlow Probability и указать её в аргументах функции.

В каких метриках у нас наблюдается глобальный тренд?

- **Количество лайков**
- **DAU**
- Уникальные просматриваемые посты
- CTR

Последний пункт, о котором можно поговорить, касается DAU. Есть одна особенность данных, которая особенно ярко выражается именно в этой метрике и может плохо повлиять на качество предсказаний модели. Какая?

- Недельная сезонность
- Провал, связанный с незаходом в приложение
- **Пик, связанный с рекламной кампанией**
- Высокий уровень шума в метрике

Давайте наконец применим CausalImpact! Возьмём следующие метрики с дневным временным разрешением: DAU, CTR, число просмотров, новые посты (учтите, что один и тот же пост может просматриваться в разные дни!), уникальные просматриваемые посты.

# Библиотеки

```python
import pandas as pd
import pandahouse as ph
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
import tensorflow_probability as tfp

from causalimpact import CausalImpact
```

# Задание 1

Поддерживать и стимулировать пользовательскую активность – важная задача для продукта, подобного нашему. Для этого наша команда маркетологов решила организовать флэшмоб в ленте новостей: участники должны сделать пост, где они рассказывают какой-то интересный факт о себе, и опубликовать его с хэштегом. Три поста, собравших наибольшее число лайков, получают призы.

Флэшмоб проходил с 2025-01-17 по 2025-01-23. Ваша задача как аналитика – оценить эффективность этого мероприятия.

выгрузим данные, уже с необходимыми нам метриками, на которые мы хотим посмотреть


```python
connection = {'host': 'https://clickhouse.lab.karpov.courses',
'database':'simulator_20250120',
'user':'student',
'password':'dpo_python_2020'}
```


```python
q = '''
SELECT 
    toDate(time) AS date,
    COUNT(post_id) AS count_of_posts,
    COUNT(DISTINCT user_id) AS users_DAU,
    countIf(action = 'view') AS views,
    countIf(action = 'like')/countIf(action = 'view') CTR
FROM {db}.feed_actions 
GROUP BY date
ORDER BY 
    date
'''
```


```python
df = ph.read_clickhouse(q, connection=connection)
df.set_index('date', inplace = True, drop = True)
df.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count_of_posts</th>
      <th>users_DAU</th>
      <th>views</th>
      <th>CTR</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-07</th>
      <td>9163</td>
      <td>878</td>
      <td>7603</td>
      <td>0.205182</td>
    </tr>
    <tr>
      <th>2024-12-08</th>
      <td>75586</td>
      <td>2238</td>
      <td>62770</td>
      <td>0.204174</td>
    </tr>
    <tr>
      <th>2024-12-09</th>
      <td>126484</td>
      <td>3105</td>
      <td>105526</td>
      <td>0.198605</td>
    </tr>
    <tr>
      <th>2024-12-10</th>
      <td>142796</td>
      <td>3721</td>
      <td>118996</td>
      <td>0.200007</td>
    </tr>
    <tr>
      <th>2024-12-11</th>
      <td>208958</td>
      <td>4617</td>
      <td>174454</td>
      <td>0.197783</td>
    </tr>
  </tbody>
</table>
</div>



# 1.8  DAU

Определим даты для пре-периода и пост-периода.


```python
pre_period = [pd.Timestamp('2024-12-07'),pd.Timestamp('2025-01-16')]
```


```python
post_period = [pd.Timestamp('2025-01-17'),pd.Timestamp('2025-01-24')]
```

Запустим нашу модель и посмотри на результаты.


```python
dau_impact = CausalImpact(pd.Series(df.users_DAU), pre_period, post_period)
```


```python
print(dau_impact.summary())
```

    Posterior Inference {Causal Impact}
                              Average            Cumulative
    Actual                    16070.5            128564.0
    Prediction (s.d.)         15033.16 (1112.87) 120265.3 (8902.94)
    95% CI                    [12916.6, 17278.96][103332.8, 138231.67]
    
    Absolute effect (s.d.)    1037.34 (1112.87)  8298.7 (8902.94)
    95% CI                    [-1208.46, 3153.9] [-9667.67, 25231.2]
    
    Relative effect (s.d.)    6.9% (7.4%)        6.9% (7.4%)
    95% CI                    [-8.04%, 20.98%]   [-8.04%, 20.98%]
    
    Posterior tail-area probability p: 0.19
    Posterior prob. of a causal effect: 81.42%
    
    For more details run the command: print(impact.summary('report'))
    


```python
dau_impact.plot()
```


    
![png](Lesson%205.1%20_files/Lesson%205.1%20_18_0.png)
    


DAU значимо не изменилось. Средняя величина абсолютного эффекта - примерно 1000

# CTR

Определим даты для пре-периода и пост-периода


```python
CTR_impact = CausalImpact(pd.Series(df.CTR), pre_period, post_period)
```

Запустим нашу модель и посмотри на результаты


```python
print(CTR_impact.summary())
```

    Posterior Inference {Causal Impact}
                              Average            Cumulative
    Actual                    0.21               1.7
    Prediction (s.d.)         0.21 (0.0)         1.66 (0.02)
    95% CI                    [0.2, 0.21]        [1.63, 1.69]
    
    Absolute effect (s.d.)    0.01 (0.0)         0.05 (0.02)
    95% CI                    [0.0, 0.01]        [0.02, 0.08]
    
    Relative effect (s.d.)    2.83% (0.93%)      2.83% (0.93%)
    95% CI                    [1.04%, 4.7%]      [1.04%, 4.7%]
    
    Posterior tail-area probability p: 0.0
    Posterior prob. of a causal effect: 99.9%
    
    For more details run the command: print(impact.summary('report'))
    


```python
CTR_impact.plot()
```


    
![png](Lesson%205.1%20_files/Lesson%205.1%20_25_0.png)
    


CTR значимо, но незначительно вырос. Средняя величина абсолютного эффекта - примерно 0.01.

# Просмотры

Определим даты для пре-периода и пост-периода


```python
views_impact = CausalImpact(pd.Series(df.views), pre_period, post_period)
```

Запустим нашу модель и посмотри на результаты


```python
print(views_impact.summary())
```

    Posterior Inference {Causal Impact}
                              Average            Cumulative
    Actual                    1030725.5          8245804.0
    Prediction (s.d.)         363840.34 (48737.74)2910722.75 (389901.96)
    95% CI                    [270353.05, 461401.54][2162824.42, 3691212.32]
    
    Absolute effect (s.d.)    666885.12 (48737.74)5335081.0 (389901.96)
    95% CI                    [569323.96, 760372.45][4554591.68, 6082979.58]
    
    Relative effect (s.d.)    183.29% (13.4%)    183.29% (13.4%)
    95% CI                    [156.48%, 208.99%] [156.48%, 208.99%]
    
    Posterior tail-area probability p: 0.0
    Posterior prob. of a causal effect: 100.0%
    
    For more details run the command: print(impact.summary('report'))
    


```python
views_impact.plot()
```


    
![png](Lesson%205.1%20_files/Lesson%205.1%20_32_0.png)
    


Число просмотров значимо выросло. Средняя величина абсолютного эффекта - примерно 666 885.

# Число новых постов

Извлечем необходимые нам данные


```python
q2 = '''
SELECT 
    publish_date AS date,
    COUNT(DISTINCT post_id) AS new_posts
FROM (
    -- Определяем дату первой активности (публикации) для каждого поста
    SELECT 
        post_id, 
        MIN(toDate(time)) AS publish_date
    FROM {db}.feed_actions
    WHERE action = 'view'  -- Учитываем только просмотры как первую активность
    GROUP BY post_id
) AS post_first_dates
GROUP BY publish_date
ORDER BY publish_date;
'''
```


```python
new_posts = ph.read_clickhouse(q2, connection=connection)
new_posts.set_index('date', inplace = True, drop = True)
new_posts 
```

Определим даты для пре-периода и пост-периода


```python
new_posts_impact = CausalImpact(pd.Series(new_posts.new_posts), pre_period, post_period)
```

Запустим нашу модель и посмотри на результаты


```python
print(new_posts_impact.summary())
```

    Posterior Inference {Causal Impact}
                              Average            Cumulative
    Actual                    76.75              614.0
    Prediction (s.d.)         72.02 (4.54)       576.14 (36.3)
    95% CI                    [63.59, 81.37]     [508.7, 650.98]
    
    Absolute effect (s.d.)    4.73 (4.54)        37.86 (36.3)
    95% CI                    [-4.62, 13.16]     [-36.98, 105.3]
    
    Relative effect (s.d.)    6.57% (6.3%)       6.57% (6.3%)
    95% CI                    [-6.42%, 18.28%]   [-6.42%, 18.28%]
    
    Posterior tail-area probability p: 0.15
    Posterior prob. of a causal effect: 85.21%
    
    For more details run the command: print(impact.summary('report'))
    


```python
new_posts_impact.plot()
```


    
![png](Lesson%205.1%20_files/Lesson%205.1%20_42_0.png)
    


Число новых постов значимо не изменилось. Средняя величина абсолютного эффекта - примерно 5.

#  Число уникальных просматриваемых постов

Извлечем необходимые нам данные


```python
q3 = '''
SELECT 
    toDate(time) AS date,
    COUNT(DISTINCT post_id) AS unique_viewed_posts
FROM {db}.feed_actions
WHERE action = 'view'
GROUP BY date
ORDER BY date;
'''
```


```python
unique_posts = ph.read_clickhouse(q3, connection=connection)
unique_posts.set_index('date', inplace = True, drop = True)
unique_posts
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_viewed_posts</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-07</th>
      <td>89</td>
    </tr>
    <tr>
      <th>2024-12-08</th>
      <td>178</td>
    </tr>
    <tr>
      <th>2024-12-09</th>
      <td>191</td>
    </tr>
    <tr>
      <th>2024-12-10</th>
      <td>211</td>
    </tr>
    <tr>
      <th>2024-12-11</th>
      <td>214</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2025-02-09</th>
      <td>200</td>
    </tr>
    <tr>
      <th>2025-02-10</th>
      <td>200</td>
    </tr>
    <tr>
      <th>2025-02-11</th>
      <td>213</td>
    </tr>
    <tr>
      <th>2025-02-12</th>
      <td>201</td>
    </tr>
    <tr>
      <th>2025-02-13</th>
      <td>199</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 1 columns</p>
</div>



Определим даты для пре-периода и пост-периода


```python
unique_posts_impact = CausalImpact(pd.Series(unique_posts.unique_viewed_posts), pre_period, post_period)
```

Запустим нашу модель и посмотри на результаты


```python
print(unique_posts_impact.summary())
```

    Posterior Inference {Causal Impact}
                              Average            Cumulative
    Actual                    281.62             2253.0
    Prediction (s.d.)         197.33 (8.61)      1578.66 (68.88)
    95% CI                    [179.6, 213.35]    [1436.83, 1706.83]
    
    Absolute effect (s.d.)    84.29 (8.61)       674.34 (68.88)
    95% CI                    [68.27, 102.02]    [546.17, 816.17]
    
    Relative effect (s.d.)    42.72% (4.36%)     42.72% (4.36%)
    95% CI                    [34.6%, 51.7%]     [34.6%, 51.7%]
    
    Posterior tail-area probability p: 0.0
    Posterior prob. of a causal effect: 100.0%
    
    For more details run the command: print(impact.summary('report'))
    


```python
unique_posts_impact.plot()
```


    
![png](Lesson%205.1%20_files/Lesson%205.1%20_52_0.png)
    


Число уникальных просматриваемых постов значимо выросло. Средняя величина абсолютного эффекта - примерно 84.
   
# Задание 2

# Библиотеки
```python
import pandas as pd
import pandahouse as ph
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import numpy as np

import orbit #общий пакет
from orbit.models import DLT #один из вариантов модели
from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components #для рисования предсказаний

from orbit.diagnostics.backtest import BackTester #основной класс для бэктестинга 
from orbit.utils.params_tuning import grid_search_orbit #для подбора оптимальных параметров

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az #это полезно для визуализации MCMC
```

# Задание 2

Чем активнее наши пользователи – тем выше нагрузка на сервера. И в последнее время нам всё чаще приходят жалобы, что приложение подвисает. Звучит как задача для девопсов и инженеров! От вас тоже попросили внести свой вклад в задачу – спрогнозировать, как изменится активность пользователей в течение ближайшего месяца. Давайте попробуем это сделать!

### **2.1 Выберите основную метрику, которую вы планируете прогнозировать. Обоснуйте, почему именно она. Какое временное разрешение вы возьмёте? Будут ли какие-то дополнительные регрессоры, которые вы включите в модель?**

Выгрузим данные

```python
connection = {'host': 'https://clickhouse.lab.karpov.courses',
'database':'simulator_20250120',
'user':'student',
'password':'dpo_python_2020'}
```


```python
q = '''
SELECT toStartOfHour(time) AS hour_time,
       COUNT(DISTINCT user_id) AS users,
       COUNT(action) AS actions

FROM {db}.feed_actions
WHERE toStartOfHour(time) <= '2025-02-15'
GROUP BY hour_time
ORDER BY hour_time DESC
'''
```


```python
df = ph.read_clickhouse(q, connection=connection)
df
#df.set_index('hour_time', inplace = True, drop = True)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>users</th>
      <th>actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-02-15 00:00:00</td>
      <td>1324</td>
      <td>29971</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-02-14 23:00:00</td>
      <td>1598</td>
      <td>36794</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-02-14 22:00:00</td>
      <td>1836</td>
      <td>40963</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-02-14 21:00:00</td>
      <td>2018</td>
      <td>42992</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-02-14 20:00:00</td>
      <td>2215</td>
      <td>42308</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>2024-12-07 04:00:00</td>
      <td>14</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>2024-12-07 03:00:00</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>2024-12-07 02:00:00</td>
      <td>16</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>2024-12-07 01:00:00</td>
      <td>26</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>2024-12-07 00:00:00</td>
      <td>14</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
<p>1681 rows × 3 columns</p>
</div>




```python
plt.figure(figsize=(16, 6))
sns.lineplot(x="hour_time", y="actions", data=df)
```




    <Axes: xlabel='hour_time', ylabel='actions'>




    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_10_1.png)
    



```python
plt.figure(figsize=(16, 6))
sns.lineplot(x="hour_time", y="users", data=df)
```

    <Axes: xlabel='hour_time', ylabel='users'>
 
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_11_1.png)
    


Посмоторим на корреляцию между DAU и количеством действий совершаемых пользователями за день, в нашем приложении новостей


```python
df.drop("hour_time", axis=1).corr()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>users</th>
      <th>actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>users</th>
      <td>1.000000</td>
      <td>0.823492</td>
    </tr>
    <tr>
      <th>actions</th>
      <td>0.823492</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Что же, мы наблюдаем вполне существенную корреляцию между выбранными переменными. Поэтому основной метрикой для прогнозирования выберем количество лайков/просмотров в день (actions), так как они в полной мере отражают активность пользователей в сервисе. Пре период возьмем все время существования нашей новостной ленты, то есть с 07.12.2024. При построении моделей с регрессорами, будем использовать метрику DAU. Я считаю что, показатель количества уникальных пользователей посетивших приложение, вкупе с основной метрикой, помогут построить максимально точную модель, на основе которой можно будет будет определить загрузку серверов нашего приложения ближайший месяц и избежать багов и прочих неприятностей.

### 2.2 Построение моделей

### 2.2.1 Damped Local Trend (MAP) без регрессоров


```python
df['hour_time'] = pd.to_datetime(df['hour_time'])
```


```python
df = df.sort_values(by='hour_time').reset_index(drop=True)
```


```python
df = df.drop_duplicates(subset=['hour_time'])
```


```python
actions_fit_MAP = df.drop("users", axis=1)
```


```python
dlt_MAP = DLT(response_col="actions", #название колонки с метрикой
          date_col="hour_time", #название колонки с датами-временем
          seasonality=24, #длина периода сезонности
          estimator="stan-map", #алгоритм оценки
          n_bootstrap_draws=1000)
```

    2025-02-16 15:46:49 - orbit - INFO - Optimizing (PyStan) with algorithm: LBFGS.
    


```python
dlt_MAP.fit(actions_fit_MAP)
```

    2025-02-16 15:46:49 - orbit - INFO - First time in running stan model:dlt. Expect 3 - 5 minutes for compilation.
    




    <orbit.forecaster.map.MAPForecaster at 0x7f630e8ec070>




```python
future_df_MAP = dlt_MAP.make_future_df(periods=24*30) #горизонт будет 30 дней - то есть 30 раз по 24 часа
future_df_MAP.tail()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>715</th>
      <td>2025-03-16 20:00:00</td>
    </tr>
    <tr>
      <th>716</th>
      <td>2025-03-16 21:00:00</td>
    </tr>
    <tr>
      <th>717</th>
      <td>2025-03-16 22:00:00</td>
    </tr>
    <tr>
      <th>718</th>
      <td>2025-03-16 23:00:00</td>
    </tr>
    <tr>
      <th>719</th>
      <td>2025-03-17 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
predicted_df_MAP = dlt_MAP.predict(df=future_df_MAP)
predicted_df_MAP.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>prediction_5</th>
      <th>prediction</th>
      <th>prediction_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-02-15 01:00:00</td>
      <td>15622.434825</td>
      <td>19062.873082</td>
      <td>23168.755763</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-02-15 02:00:00</td>
      <td>8919.987308</td>
      <td>13900.402142</td>
      <td>18800.576534</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-02-15 03:00:00</td>
      <td>2805.431827</td>
      <td>8600.071543</td>
      <td>14387.686571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-02-15 04:00:00</td>
      <td>1286.079766</td>
      <td>7514.941219</td>
      <td>13866.168609</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-02-15 05:00:00</td>
      <td>5740.351362</td>
      <td>13306.511114</td>
      <td>20803.532559</td>
    </tr>
  </tbody>
</table>
</div>


```python
_ = plot_predicted_data(actions_fit_MAP, predicted_df_MAP, "hour_time", 'actions', title='Prediction with DLT')
```    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_25_1.png)


```python
_ = plot_predicted_components(predicted_df_MAP, "hour_time", plot_components=['prediction', 'trend', 'seasonality'])
```
   
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_26_1.png)
    


Выберем скользящий рзмер тренировочных данных, так как наши данные волатильны, ввиду роста приложения, рекламных компаний и так как основная информация о предсказании находится в ближайшем прошлом.


```python
bt_roll_MAP = BackTester(
    model=dlt_MAP,
    df=actions_fit_MAP,
    min_train_len=24*14,
    incremental_len=24*2,
    forecast_len=24,
    window_type="rolling",
)
```

```python
#bt_roll_MAP.plot_scheme()
```


```python
bt_roll_MAP.fit_predict() #обучаем
bt_roll_MAP.score() #выводим метрики
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric_name</th>
      <th>metric_values</th>
      <th>is_training_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape</td>
      <td>1.796916e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wmape</td>
      <td>1.653128e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mape</td>
      <td>1.803363e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mse</td>
      <td>6.136099e+07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mae</td>
      <td>5.075370e+03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmsse</td>
      <td>1.798488e+00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Вывод: наблюдаем, что из-за выбросов в виде рекламного ивента, самая простая модель, без регрессоров, построила довольно посредственное предсказание с большим доверительным интервалом. 

### 2.2.2 Damped Local Trend (MCMC) без регрессоров


```python
actions_fit_MCMC = df.drop("users", axis=1)
```


```python
dlt_MCMC = DLT(seasonality=24, response_col="actions", date_col="hour_time", 
               estimator='stan-mcmc', #новый алгоритм оценки
               num_warmup=2000, #время "разогрева"
               num_sample=1000) #время сэмплирования
```


```python
dlt_MCMC.fit(actions_fit_MCMC)
```

```python
future_df_MCMC = dlt_MCMC.make_future_df(periods=24*30) #горизонт будет 30 дней - то есть 30 раз по 24 часа
future_df_MCMC.tail()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>715</th>
      <td>2025-03-16 20:00:00</td>
    </tr>
    <tr>
      <th>716</th>
      <td>2025-03-16 21:00:00</td>
    </tr>
    <tr>
      <th>717</th>
      <td>2025-03-16 22:00:00</td>
    </tr>
    <tr>
      <th>718</th>
      <td>2025-03-16 23:00:00</td>
    </tr>
    <tr>
      <th>719</th>
      <td>2025-03-17 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
predicted_df_MCMC = dlt_MCMC.predict(df=future_df_MCMC)
predicted_df_MCMC.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>prediction_5</th>
      <th>prediction</th>
      <th>prediction_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-02-15 01:00:00</td>
      <td>12949.004543</td>
      <td>19051.944404</td>
      <td>23047.310963</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-02-15 02:00:00</td>
      <td>5453.249511</td>
      <td>13089.951124</td>
      <td>18342.163297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-02-15 03:00:00</td>
      <td>-3024.980007</td>
      <td>7344.371611</td>
      <td>13612.618919</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-02-15 04:00:00</td>
      <td>-5142.529728</td>
      <td>5784.826683</td>
      <td>12551.056881</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-02-15 05:00:00</td>
      <td>-858.679040</td>
      <td>11648.554963</td>
      <td>18597.176072</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = plot_predicted_data(actions_fit_MCMC, predicted_df_MCMC, "hour_time", 'actions', title='Prediction with DLT')
```

    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_38_1.png)
    



```python
_ = plot_predicted_components(predicted_df_MCMC, "hour_time", plot_components=['prediction', 'trend', 'seasonality'])
```

    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_39_1.png)
    



```python
bt_roll_MCMC = BackTester(
    model=dlt_MCMC,
    df=actions_fit_MCMC,
    min_train_len=24*7,
    incremental_len=24,
    forecast_len=24,
    window_type="rolling",
)
```


```python
#bt_roll_MCMC.plot_scheme()
```


```python
bt_roll_MCMC.fit_predict() #обучаем
```  

```python
bt_roll_MCMC.score() #выводим метрики
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric_name</th>
      <th>metric_values</th>
      <th>is_training_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape</td>
      <td>4.229238e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wmape</td>
      <td>3.462096e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mape</td>
      <td>3.740291e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mse</td>
      <td>1.761574e+08</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mae</td>
      <td>9.989581e+03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmsse</td>
      <td>3.055412e+00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
params_MCMC = dlt_MCMC.get_posterior_samples(permute=False) #достаём информацию о параметрах
```


```python
az.plot_trace(params_MCMC, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```

    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_45_1.png)
    


Наблюдаем, что модель DLT с алгоритмом MCMC, оказалась очень чувствительна выбросам и показала ухудшение оценки ее предсказательной способности, по сравнению с алгоритмом MAP. В таком виде применять ее не стоит, необходимо настроить параметры алгоритма, таким образом, чтобы он занижал значения выбросов.

### 2.2.3 Damped Local Trend (MCMC) без регрессоров, с настройкой параметров алгоритма


```python
actions_fit_MCMC_par = df.drop("users", axis=1)
```


```python
dlt_MCMC_par = DLT(
    seasonality=24,
    response_col="actions",
    date_col="hour_time",
    estimator="stan-mcmc",
    num_warmup=2000,
    num_sample=1000,
    level_sm_input=0.3,
    chains=4,  # Передаём chains отдельно
    stan_mcmc_args={"adapt_delta": 0.9},  
)

```


```python
dlt_MCMC_par.fit(actions_fit_MCMC_par)
```

```python
future_df_MCMC_par = dlt_MCMC_par.make_future_df(periods=24*30) #горизонт будет 30 дней - то есть 30 раз по 24 часа
future_df_MCMC_par.tail()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>715</th>
      <td>2025-03-16 20:00:00</td>
    </tr>
    <tr>
      <th>716</th>
      <td>2025-03-16 21:00:00</td>
    </tr>
    <tr>
      <th>717</th>
      <td>2025-03-16 22:00:00</td>
    </tr>
    <tr>
      <th>718</th>
      <td>2025-03-16 23:00:00</td>
    </tr>
    <tr>
      <th>719</th>
      <td>2025-03-17 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
predicted_df_MCMC_par = dlt_MCMC_par.predict(df=future_df_MCMC_par)
predicted_df_MCMC_par.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>prediction_5</th>
      <th>prediction</th>
      <th>prediction_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-02-15 01:00:00</td>
      <td>18535.760050</td>
      <td>22444.068648</td>
      <td>26703.050796</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-02-15 02:00:00</td>
      <td>11443.813314</td>
      <td>15831.050083</td>
      <td>20667.885181</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-02-15 03:00:00</td>
      <td>5818.409727</td>
      <td>10413.075897</td>
      <td>15118.415603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-02-15 04:00:00</td>
      <td>4927.085025</td>
      <td>9687.198732</td>
      <td>14583.321975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-02-15 05:00:00</td>
      <td>10514.145658</td>
      <td>15872.911033</td>
      <td>20858.174408</td>
    </tr>
  </tbody>
</table>
</div>

```python
_ = plot_predicted_data(actions_fit_MCMC_par, predicted_df_MCMC_par, "hour_time", 'actions', title='Prediction with DLT')
```

    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_53_1.png)
    



```python
_ = plot_predicted_components(predicted_df_MCMC_par, "hour_time", plot_components=['prediction', 'trend', 'seasonality'])
```

   
    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_54_1.png)
    



```python
bt_roll_MCMC_par = BackTester(
    model=dlt_MCMC_par,
    df=actions_fit_MCMC_par,
    min_train_len=24*7,
    incremental_len=24,
    forecast_len=24,
    window_type="rolling",
)
```


```python
bt_roll_MCMC_par.plot_scheme()
```


    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_56_0.png)
    



```python
bt_roll_MCMC_par.fit_predict() 
```
                                                                                                                                                                                                                                                                                                                                  


```python
bt_roll_MCMC_par.score()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric_name</th>
      <th>metric_values</th>
      <th>is_training_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape</td>
      <td>1.723776e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wmape</td>
      <td>1.665540e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mape</td>
      <td>1.819710e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mse</td>
      <td>5.906530e+07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mae</td>
      <td>4.805772e+03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmsse</td>
      <td>1.769235e+00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
params_MCMC_par = dlt_MCMC_par.get_posterior_samples(permute=False) #достаём информацию о параметрах
```


```python
az.plot_trace(params_MCMC_par, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```
    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_60_1.png)
    


Видим определенные улучшения на графике, но предсказательная способность модели существенно не изменилась.

### 2.2.4 Создадим регрессор


```python
users = df.drop("actions", axis=1)
```


```python
users = users.sort_values(by='hour_time').reset_index(drop=True)
```


```python
dlt_users_map = DLT(response_col="users", #название колонки с метрикой
          date_col="hour_time", #название колонки с датами-временем
          seasonality=24, #длина периода сезонности
          estimator="stan-map", #алгоритм оценки
          n_bootstrap_draws=1000)
```

    2025-02-16 16:14:17 - orbit - INFO - Optimizing (PyStan) with algorithm: LBFGS.
    


```python
dlt_users_map.fit(users)
```




    <orbit.forecaster.map.MAPForecaster at 0x7f6246181430>




```python
future_df_users_map = dlt_users_map.make_future_df(periods=24*30) #горизонт будет 30 дней - то есть 30 раз по 24 час
```


```python
predicted_df_users_map = dlt_users_map.predict(df=future_df_MAP)
predicted_df_users_map
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>prediction_5</th>
      <th>prediction</th>
      <th>prediction_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-02-15 01:00:00</td>
      <td>904.105321</td>
      <td>1018.633557</td>
      <td>1136.712403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-02-15 02:00:00</td>
      <td>678.704847</td>
      <td>803.449784</td>
      <td>919.945830</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-02-15 03:00:00</td>
      <td>435.691361</td>
      <td>558.474011</td>
      <td>676.188724</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-02-15 04:00:00</td>
      <td>436.082605</td>
      <td>554.841238</td>
      <td>675.173623</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-02-15 05:00:00</td>
      <td>701.431953</td>
      <td>826.308465</td>
      <td>943.637461</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>715</th>
      <td>2025-03-16 20:00:00</td>
      <td>1501.485166</td>
      <td>2100.648922</td>
      <td>2747.396775</td>
    </tr>
    <tr>
      <th>716</th>
      <td>2025-03-16 21:00:00</td>
      <td>1330.654942</td>
      <td>1898.062149</td>
      <td>2518.273272</td>
    </tr>
    <tr>
      <th>717</th>
      <td>2025-03-16 22:00:00</td>
      <td>1113.083083</td>
      <td>1689.576376</td>
      <td>2317.039009</td>
    </tr>
    <tr>
      <th>718</th>
      <td>2025-03-16 23:00:00</td>
      <td>916.447507</td>
      <td>1521.837603</td>
      <td>2162.181244</td>
    </tr>
    <tr>
      <th>719</th>
      <td>2025-03-17 00:00:00</td>
      <td>647.300183</td>
      <td>1271.194830</td>
      <td>1879.423424</td>
    </tr>
  </tbody>
</table>
<p>720 rows × 4 columns</p>
</div>




```python
pred_users = predicted_df_users_map[['prediction', 'hour_time']]
pred_users = predicted_df_users_map.rename(columns={'prediction': 'users'})
pred_users = pred_users.drop(["prediction_5", 'prediction_95'], axis=1)
pred_users
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>users</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-02-15 01:00:00</td>
      <td>1018.633557</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-02-15 02:00:00</td>
      <td>803.449784</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-02-15 03:00:00</td>
      <td>558.474011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-02-15 04:00:00</td>
      <td>554.841238</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-02-15 05:00:00</td>
      <td>826.308465</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>715</th>
      <td>2025-03-16 20:00:00</td>
      <td>2100.648922</td>
    </tr>
    <tr>
      <th>716</th>
      <td>2025-03-16 21:00:00</td>
      <td>1898.062149</td>
    </tr>
    <tr>
      <th>717</th>
      <td>2025-03-16 22:00:00</td>
      <td>1689.576376</td>
    </tr>
    <tr>
      <th>718</th>
      <td>2025-03-16 23:00:00</td>
      <td>1521.837603</td>
    </tr>
    <tr>
      <th>719</th>
      <td>2025-03-17 00:00:00</td>
      <td>1271.194830</td>
    </tr>
  </tbody>
</table>
<p>720 rows × 2 columns</p>
</div>



### 2.2.5 Damped Local Trend (MAP) с регрессором


```python
reg_map = df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>users</th>
      <th>actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-12-07 00:00:00</td>
      <td>14</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-12-07 01:00:00</td>
      <td>26</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-12-07 02:00:00</td>
      <td>16</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-12-07 03:00:00</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-12-07 04:00:00</td>
      <td>14</td>
      <td>18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>2025-02-14 20:00:00</td>
      <td>2215</td>
      <td>42308</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>2025-02-14 21:00:00</td>
      <td>2018</td>
      <td>42992</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>2025-02-14 22:00:00</td>
      <td>1836</td>
      <td>40963</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>2025-02-14 23:00:00</td>
      <td>1598</td>
      <td>36794</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>2025-02-15 00:00:00</td>
      <td>1324</td>
      <td>29971</td>
    </tr>
  </tbody>
</table>
<p>1681 rows × 3 columns</p>
</div>




```python
map_date = pd.concat([reg_map, pred_users], ignore_index=True)
map_date
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>users</th>
      <th>actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-12-07 00:00:00</td>
      <td>14.000000</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-12-07 01:00:00</td>
      <td>26.000000</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-12-07 02:00:00</td>
      <td>16.000000</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-12-07 03:00:00</td>
      <td>11.000000</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-12-07 04:00:00</td>
      <td>14.000000</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2396</th>
      <td>2025-03-16 20:00:00</td>
      <td>2100.648922</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2397</th>
      <td>2025-03-16 21:00:00</td>
      <td>1898.062149</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2398</th>
      <td>2025-03-16 22:00:00</td>
      <td>1689.576376</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2399</th>
      <td>2025-03-16 23:00:00</td>
      <td>1521.837603</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2400</th>
      <td>2025-03-17 00:00:00</td>
      <td>1271.194830</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2401 rows × 3 columns</p>
</div>




```python
train_dat = reg_map.astype({"actions":"float64"})
test_dat = pred_users.astype({"users":"int64"})
```


```python
reg_map_model = DLT(response_col="actions",
                date_col="hour_time", 
                seasonality=24,
                estimator="stan-map", 
                n_bootstrap_draws=1000, 
                regressor_col=["users"], #наша колонка с регрессором! Должна быть списком
               ) 

reg_map_model.fit(train_dat)
```


```python
predicted_df_reg_map = reg_map_model.predict(df=test_dat)
_ = plot_predicted_data(map_date, predicted_df_reg_map, "hour_time", 'actions', title='Prediction with DLT')
```

   
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_75_1.png)
    



```python
rf = df
```


```python
rf['users'] = rf['users'].astype('float64')
rf['actions'] = rf['actions'].astype('float64')
rf
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>users</th>
      <th>actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-12-07 00:00:00</td>
      <td>14.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-12-07 01:00:00</td>
      <td>26.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-12-07 02:00:00</td>
      <td>16.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-12-07 03:00:00</td>
      <td>11.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-12-07 04:00:00</td>
      <td>14.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>2025-02-14 20:00:00</td>
      <td>2215.0</td>
      <td>42308.0</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>2025-02-14 21:00:00</td>
      <td>2018.0</td>
      <td>42992.0</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>2025-02-14 22:00:00</td>
      <td>1836.0</td>
      <td>40963.0</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>2025-02-14 23:00:00</td>
      <td>1598.0</td>
      <td>36794.0</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>2025-02-15 00:00:00</td>
      <td>1324.0</td>
      <td>29971.0</td>
    </tr>
  </tbody>
</table>
<p>1681 rows × 3 columns</p>
</div>




```python
bt_roll_reg_map = BackTester(
    model=reg_map_model,
    df=rf,
    min_train_len=24*14,
    incremental_len=24*7,
    forecast_len=24,
    window_type="rolling",
)
```


```python
bt_roll_reg_map.fit_predict() #обучаем
```


```python
bt_roll_reg_map.score() #выводим метрики
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric_name</th>
      <th>metric_values</th>
      <th>is_training_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape</td>
      <td>1.805347e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wmape</td>
      <td>1.942598e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mape</td>
      <td>2.270895e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mse</td>
      <td>9.002583e+07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mae</td>
      <td>5.593552e+03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmsse</td>
      <td>2.183759e+00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2.6 Damped Local Trend (MСMC) с регрессором


```python
reg_mcmc = df
```


```python
mcmc_date = pd.concat([reg_mcmc, pred_users], ignore_index=True)
mcmc_date
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_time</th>
      <th>users</th>
      <th>actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-12-07 00:00:00</td>
      <td>14.000000</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-12-07 01:00:00</td>
      <td>26.000000</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-12-07 02:00:00</td>
      <td>16.000000</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-12-07 03:00:00</td>
      <td>11.000000</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-12-07 04:00:00</td>
      <td>14.000000</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2396</th>
      <td>2025-03-16 20:00:00</td>
      <td>2100.648922</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2397</th>
      <td>2025-03-16 21:00:00</td>
      <td>1898.062149</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2398</th>
      <td>2025-03-16 22:00:00</td>
      <td>1689.576376</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2399</th>
      <td>2025-03-16 23:00:00</td>
      <td>1521.837603</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2400</th>
      <td>2025-03-17 00:00:00</td>
      <td>1271.194830</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2401 rows × 3 columns</p>
</div>




```python
train_dat = reg_mcmc
test_dat= pred_users.astype({"users":"int64"})
```


```python
reg_mcmc_model = DLT(response_col="actions",
                date_col="hour_time", 
                seasonality=24,
                estimator="stan-mcmc", 
                n_bootstrap_draws=1000, 
                regressor_col=["users"], #наша колонка с регрессором! Должна быть списком
               ) 

reg_mcmc_model.fit(train_dat)
```

```python
predicted_df_reg_mcmc = reg_mcmc_model.predict(df=test_dat)
_ = plot_predicted_data(mcmc_date, predicted_df_reg_mcmc, "hour_time", 'actions', title='Prediction with DLT')
```

   
    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_86_1.png)
    



```python
bt_roll_reg_mcmc = BackTester(
    model=reg_mcmc_model,
    df=mcmc_date,
    min_train_len=24*14,
    incremental_len=24*2,
    forecast_len=24,
    window_type="rolling",
)
```


```python
bt_roll_reg_mcmc.fit_predict() #обучаем
```


```python
params_MCMC_1 = reg_mcmc_model.get_posterior_samples(permute=False) #достаём информацию о параметрах
az.plot_trace(params_MCMC_1, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```


```python
bt_roll_reg_mcmc.score() #выводим метрики
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric_name</th>
      <th>metric_values</th>
      <th>is_training_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape</td>
      <td>1.504933e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wmape</td>
      <td>1.450757e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mape</td>
      <td>1.614939e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mse</td>
      <td>3.793239e+07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mae</td>
      <td>4.453902e+03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmsse</td>
      <td>1.408206e+00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2.7 Damped Local Trend (MСMC) с регрессором и априорным распределением 


```python
reg_MCMC_model_apr = DLT(response_col="actions",
                date_col="hour_time", 
                seasonality=24,
                estimator="stan-mcmc", 
                n_bootstrap_draws=1000, 
                regressor_col=["users"], #наша колонка с регрессором! Должна быть списком
                regressor_sign=["+"], #допустим, мы считаем, что связь обязательно положительная
                regressor_beta_prior=[0.3], #пусть мы думаем, что истинное значение коэффициента - вот такое
                regressor_sigma_prior=[0.1] #и зададим уровень уверенности
               ) 

reg_MCMC_model_apr.fit(train_dat)
```

```python
predicted_df_reg_MCMC = reg_MCMC_model_apr.predict(df=test_dat)
_ = plot_predicted_data(mcmc_date, predicted_df_reg_MCMC, "hour_time", 'actions', title='Prediction with DLT')
```

    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_93_1.png)
    



```python
bt_roll_reg_MCMC_apr = BackTester(
    model=reg_MCMC_model_apr,
    df=mcmc_date,
    min_train_len=24*14,
    incremental_len=24*2,
    forecast_len=24,
    window_type="rolling",
)
```


```python
bt_roll_reg_MCMC_apr.fit_predict() #обучаем
```

```python
params_MCMC_2 = reg_MCMC_model_apr.get_posterior_samples(permute=False) #достаём информацию о параметрах
az.plot_trace(params_MCMC_2, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```


    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_96_1.png)
    



```python
bt_roll_reg_MCMC_apr.score() #выводим метрики
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric_name</th>
      <th>metric_values</th>
      <th>is_training_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape</td>
      <td>2.437420e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wmape</td>
      <td>2.255296e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mape</td>
      <td>2.372836e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mse</td>
      <td>9.873530e+07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mae</td>
      <td>6.923879e+03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmsse</td>
      <td>2.271942e+00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2.8 Damped Local Trend (MСMC) с регрессором и регуляризацией


```python
reg_MCMC_model_rl = DLT(response_col="actions",
                date_col="hour_time", 
                seasonality=24,
                estimator="stan-mcmc", 
                n_bootstrap_draws=1000, 
                regressor_col=["users"], #наша колонка с регрессором! Должна быть списком
                regression_penalty="auto_ridge")

reg_MCMC_model_rl.fit(train_dat)
```





```python
predicted_df_reg_MCMC = reg_MCMC_model_rl.predict(df=test_dat)
_ = plot_predicted_data(mcmc_date, predicted_df_reg_MCMC, "hour_time", 'actions', title='Prediction with DLT')
```

    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_100_1.png)
    



```python
bt_roll_reg_MCMC = BackTester(
    model=reg_MCMC_model_rl,
    df=mcmc_date,
    min_train_len=24*14,
    incremental_len=24*2,
    forecast_len=24,
    window_type="rolling",
)
```


```python
bt_roll_reg_MCMC.fit_predict()
```                                                                                                                                                                                                                   
```python
params_MCMC_3 = reg_MCMC_model_rl.get_posterior_samples(permute=False) #достаём информацию о параметрах
az.plot_trace(params_MCMC_3, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```

![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_103_1.png)
    



```python
bt_roll_reg_MCMC.score()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric_name</th>
      <th>metric_values</th>
      <th>is_training_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>smape</td>
      <td>2.463281e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wmape</td>
      <td>2.692322e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mape</td>
      <td>2.970607e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mse</td>
      <td>1.137448e+08</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mae</td>
      <td>8.265571e+03</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmsse</td>
      <td>2.438523e+00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



# 2.2.9 Вывод

Выберать лучшую модель из получившихся будем по метрике RMSSE. Это показатель, который сравнивает точность модели с простым прогнозом (например, повторением последнего значения). Чем меньше RMSSE, тем лучше модель. Если RMSSE больше 1, это значит, что модель хуже простого прогноза.


```python
print(f"Модель Damped Local Trend (MAP) без регрессора (RMSSE): {bt_roll_MAP.score().loc[bt_roll_MAP.score()['metric_name'] == 'rmsse', 'metric_values'].values[0]:.4f}")
print(f"Модель Damped Local Trend (MAP) c регрессором (RMSSE): {bt_roll_reg_map.score().loc[bt_roll_reg_map.score()['metric_name'] == 'rmsse', 'metric_values'].values[0]:.4f}")
print(f"Модель Damped Local Trend (MCMC) c регрессором (RMSSE): {bt_roll_reg_mcmc.score().loc[bt_roll_reg_mcmc.score()['metric_name'] == 'rmsse', 'metric_values'].values[0]:.4f}")
print(f"Модель Damped Local Trend (MCMC) без регрессора (RMSSE): {bt_roll_MCMC.score().loc[bt_roll_MCMC.score()['metric_name'] == 'rmsse', 'metric_values'].values[0]:.4f}")
print(f"Модель Damped Local Trend (MCMC) без регрессора, с настройкой параметров алгоритма (RMSSE): {bt_roll_MCMC_par.score().loc[bt_roll_MCMC_par.score()['metric_name'] == 'rmsse', 'metric_values'].values[0]:.4f}")
print(f"Модель Damped Local Trend (MCMC)  с регрессором и регуляризацией (RMSSE): {bt_roll_reg_MCMC.score().loc[bt_roll_reg_MCMC.score()['metric_name'] == 'rmsse', 'metric_values'].values[0]:.4f}")
```

    Модель Damped Local Trend (MAP) без регрессора (RMSSE): 1.7985
    Модель Damped Local Trend (MAP) c регрессором (RMSSE): 2.1838
    Модель Damped Local Trend (MCMC) c регрессором (RMSSE): 1.4082
    Модель Damped Local Trend (MCMC) без регрессора (RMSSE): 3.0554
    Модель Damped Local Trend (MCMC) без регрессора, с настройкой параметров алгоритма (RMSSE): 1.7692
    Модель Damped Local Trend (MCMC)  с регрессором и регуляризацией (RMSSE): 2.4385
    

По итогу, можно констатировать, что у нас получились модели с довольно посредственными предсказательными способностями, ни у одной из них RMSSE не получился меньше 1. Лучшая по метрике RMSSE, это модель Damped Local Trend (MCMC) c регрессором, чей (RMSSE): 1.4082.


```python

```
