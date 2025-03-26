# Библиотеки


```python
pip install pandahouse
```

    Requirement already satisfied: pandahouse in /opt/conda/lib/python3.8/site-packages (0.2.7)
    Requirement already satisfied: toolz in /opt/conda/lib/python3.8/site-packages (from pandahouse) (0.12.0)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.8/site-packages (from pandahouse) (2.0.2)
    Requirement already satisfied: requests in /opt/conda/lib/python3.8/site-packages (from pandahouse) (2.27.1)
    Requirement already satisfied: tzdata>=2022.1 in /opt/conda/lib/python3.8/site-packages (from pandas->pandahouse) (2023.3)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.8/site-packages (from pandas->pandahouse) (2021.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.8/site-packages (from pandas->pandahouse) (2.8.2)
    Requirement already satisfied: numpy>=1.20.3 in /opt/conda/lib/python3.8/site-packages (from pandas->pandahouse) (1.23.5)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (2021.10.8)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (3.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (2.0.11)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (1.26.8)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.8/site-packages (from python-dateutil>=2.8.2->pandas->pandahouse) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


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

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_25_1.png)
    



```python
_ = plot_predicted_components(predicted_df_MAP, "hour_time", plot_components=['prediction', 'trend', 'seasonality'])
```

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
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

    2025-02-16 15:47:32 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    




    <orbit.forecaster.full_bayes.FullBayesianForecaster at 0x7f630e6bae50>




```python
future_df_MCMC = dlt_MCMC.make_future_df(periods=24*30) #горизонт будет 30 дней - то есть 30 раз по 24 часа
future_df_MCMC.tail()
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

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_38_1.png)
    



```python
_ = plot_predicted_components(predicted_df_MCMC, "hour_time", plot_components=['prediction', 'trend', 'seasonality'])
```

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
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

    2025-02-16 15:52:04 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:52:13 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:52:23 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:52:32 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:52:42 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:52:52 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:53:02 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:53:11 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:53:20 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:53:30 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:53:40 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:53:50 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:54:00 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:54:09 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:54:20 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:54:29 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:54:38 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:54:47 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:54:56 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:55:05 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:55:14 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:55:25 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:55:33 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:55:42 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:55:50 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:56:00 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:56:09 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:56:19 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:56:28 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:56:37 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:56:46 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:56:56 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:57:05 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:57:14 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:57:23 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:57:32 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:57:41 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:57:50 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:57:59 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:58:09 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:58:19 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:58:30 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:58:39 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:58:48 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:58:57 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:59:06 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:59:17 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:59:28 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:59:37 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:59:48 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 15:59:57 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:00:08 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:00:18 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:00:29 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:00:38 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:00:48 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:00:57 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:01:06 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:01:15 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:01:25 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:01:34 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:01:43 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:01:52 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    


```python
bt_roll_MCMC.score() #выводим метрики
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




    array([[<Axes: title={'center': 'obs_sigma'}>,
            <Axes: title={'center': 'obs_sigma'}>]], dtype=object)




    
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

    2025-02-16 16:02:02 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    




    <orbit.forecaster.full_bayes.FullBayesianForecaster at 0x7f62464b3cd0>




```python
future_df_MCMC_par = dlt_MCMC_par.make_future_df(periods=24*30) #горизонт будет 30 дней - то есть 30 раз по 24 часа
future_df_MCMC_par.tail()
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

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_53_1.png)
    



```python
_ = plot_predicted_components(predicted_df_MCMC_par, "hour_time", plot_components=['prediction', 'trend', 'seasonality'])
```

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
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

    2025-02-16 16:04:03 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:04:13 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:04:22 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:04:31 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:04:41 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:04:51 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:05:00 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:05:11 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:05:20 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:05:30 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:05:39 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:05:48 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:05:58 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:06:08 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:06:17 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:06:26 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:06:38 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:06:47 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:06:56 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:07:06 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:07:17 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:07:26 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:07:35 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:07:45 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:07:54 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:08:04 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:08:15 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:08:26 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:08:34 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:08:45 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:08:55 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:09:04 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:09:13 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:09:23 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:09:31 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:09:41 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:09:51 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:10:00 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:10:11 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:10:20 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:10:30 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:10:40 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:10:49 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:10:58 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:11:08 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:11:18 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:11:28 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:11:38 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:11:48 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:11:56 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:12:06 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:12:16 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:12:26 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:12:36 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:12:48 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:12:59 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:13:08 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:13:18 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:13:27 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:13:36 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:13:45 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:13:56 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:14:07 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 500 and samples(per chain): 250.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    


```python
bt_roll_MCMC_par.score()
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




    array([[<Axes: title={'center': 'obs_sigma'}>,
            <Axes: title={'center': 'obs_sigma'}>]], dtype=object)




    
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

    2025-02-16 16:36:37 - orbit - INFO - Optimizing (PyStan) with algorithm: LBFGS.
    




    <orbit.forecaster.map.MAPForecaster at 0x7f622b8b77f0>




```python
predicted_df_reg_map = reg_map_model.predict(df=test_dat)
_ = plot_predicted_data(map_date, predicted_df_reg_map, "hour_time", 'actions', title='Prediction with DLT')
```

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_75_1.png)
    



```python
rf = df
```


```python
rf['users'] = rf['users'].astype('float64')
rf['actions'] = rf['actions'].astype('float64')
rf
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

    2025-02-16 16:22:02 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    




    <orbit.forecaster.full_bayes.FullBayesianForecaster at 0x7f6233ee43a0>




```python
predicted_df_reg_mcmc = reg_mcmc_model.predict(df=test_dat)
_ = plot_predicted_data(mcmc_date, predicted_df_reg_mcmc, "hour_time", 'actions', title='Prediction with DLT')
```

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
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

    2025-02-16 17:23:39 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:23:54 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:24:09 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:24:25 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:24:39 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:24:53 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:25:10 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:25:26 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:25:42 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:25:58 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:26:13 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:26:29 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:26:44 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:26:59 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:27:15 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:27:29 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:27:45 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:27:59 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:28:14 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:28:28 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:28:47 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:28:59 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:29:14 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:29:27 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:29:42 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:29:57 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:30:12 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:30:28 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:30:45 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:31:01 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:31:15 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:31:28 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:31:40 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:31:53 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:32:03 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    /opt/conda/lib/python3.8/site-packages/orbit/template/ets.py:129: RuntimeWarning: Mean of empty slice
      ss[idx] = np.nanmean(adjusted_response[idx :: self._seasonality])
    /opt/conda/lib/python3.8/site-packages/numpy/lib/nanfunctions.py:1878: RuntimeWarning: Degrees of freedom <= 0 for slice.
      var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    2025-02-16 17:32:10 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Input In [137], in <module>
    ----> 1 bt_roll_reg_mcmc.fit_predict()
    

    File /opt/conda/lib/python3.8/site-packages/orbit/diagnostics/backtest.py:383, in BackTester.fit_predict(self)
        381 for train_df, test_df, scheme, key in splitter.split():
        382     model_copy = deepcopy(model)
    --> 383     model_copy.fit(train_df)
        384     train_predictions = model_copy.predict(train_df)
        385     test_predictions = model_copy.predict(test_df)
    

    File /opt/conda/lib/python3.8/site-packages/orbit/forecaster/full_bayes.py:36, in FullBayesianForecaster.fit(self, df, point_method, keep_samples, sampling_temperature, **kwargs)
         28 def fit(
         29     self,
         30     df,
       (...)
         34     **kwargs,
         35 ):
    ---> 36     super().fit(df, sampling_temperature=sampling_temperature, **kwargs)
         37     self._point_method = point_method
         39     if point_method is not None:
    

    File /opt/conda/lib/python3.8/site-packages/orbit/forecaster/forecaster.py:162, in Forecaster.fit(self, df, **kwargs)
        158 model_param_names = self._model.get_model_param_names()
        160 # note that estimator will search for the .stan, .pyro model file based on the
        161 # estimator type and model_name provided
    --> 162 _posterior_samples, training_metrics = estimator.fit(
        163     model_name=model_name,
        164     model_param_names=model_param_names,
        165     data_input=data_input,
        166     fitter=self._model.get_fitter(),
        167     init_values=init_values,
        168     **kwargs,
        169 )
        170 self._posterior_samples = _posterior_samples
        171 self._training_metrics = training_metrics
    

    File /opt/conda/lib/python3.8/site-packages/orbit/estimators/stan_estimator.py:140, in StanEstimatorMCMC.fit(self, model_name, model_param_names, sampling_temperature, data_input, fitter, init_values)
        137 compiled_mod = get_compiled_stan_model(model_name)
        138 # check https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.sample
        139 # for additional args
    --> 140 stan_mcmc_fit = compiled_mod.sample(
        141     data=data_input,
        142     iter_sampling=self._num_sample_per_chain,
        143     iter_warmup=self._num_warmup_per_chain,
        144     chains=self.chains,
        145     parallel_chains=self.cores,
        146     inits=init_values,
        147     seed=self.seed,
        148     **self._stan_mcmc_args,
        149 )
        151 stan_extract = stan_mcmc_fit.stan_variables()
        152 posteriors = {
        153     param: stan_extract[param] for param in model_param_names + ["loglk"]
        154 }
    

    File /opt/conda/lib/python3.8/site-packages/cmdstanpy/model.py:1201, in CmdStanModel.sample(self, data, chains, parallel_chains, threads_per_chain, seed, chain_ids, inits, iter_warmup, iter_sampling, save_warmup, thin, max_treedepth, metric, step_size, adapt_engaged, adapt_delta, adapt_init_phase, adapt_metric_window, adapt_step_size, fixed_param, output_dir, sig_figs, save_latent_dynamics, save_profile, show_progress, show_console, refresh, time_fmt, timeout, force_one_process_per_chain)
       1194 if not runset._check_retcodes():
       1195     msg = (
       1196         f'Error during sampling:\n{errors}\n'
       1197         + f'Command and output files:\n{repr(runset)}\n'
       1198         + 'Consider re-running with show_console=True if the above'
       1199         + ' output is unclear!'
       1200     )
    -> 1201     raise RuntimeError(msg)
       1202 if errors:
       1203     msg = (
       1204         f'Non-fatal error during sampling:\n{errors}\n'
       1205         + 'Consider re-running with show_console=True if the above'
       1206         + ' output is unclear!'
       1207     )
    

    RuntimeError: Error during sampling:
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Command and output files:
    RunSet: chains=4, chain_ids=[1, 2, 3, 4], num_processes=4
     cmd (chain 1):
    	['/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt', 'id=1', 'random', 'seed=8888', 'data', 'file=/tmp/tmpqqnjd0l9/lm5_gtgt.json', 'init=/tmp/tmpqqnjd0l9/2dh1azb5.json', 'output', 'file=/tmp/tmpqqnjd0l9/dltpuvy8a7n/dlt-20250216173210_1.csv', 'method=sample', 'num_samples=25', 'num_warmup=225', 'algorithm=hmc', 'adapt', 'engaged=1']
     retcodes=[1, 1, 1, 1]
     per-chain output files (showing chain 1 only):
     csv_file:
    	/tmp/tmpqqnjd0l9/dltpuvy8a7n/dlt-20250216173210_1.csv
     console_msgs (if any):
    	/tmp/tmpqqnjd0l9/dltpuvy8a7n/dlt-20250216173210_0-stdout.txt
    Consider re-running with show_console=True if the above output is unclear!



```python
params_MCMC_1 = reg_mcmc_model.get_posterior_samples(permute=False) #достаём информацию о параметрах
az.plot_trace(params_MCMC_1, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```


```python
bt_roll_reg_mcmc.score() #выводим метрики
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

    2025-02-16 16:37:28 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    




    <orbit.forecaster.full_bayes.FullBayesianForecaster at 0x7f6232fab3a0>




```python
predicted_df_reg_MCMC = reg_MCMC_model_apr.predict(df=test_dat)
_ = plot_predicted_data(mcmc_date, predicted_df_reg_MCMC, "hour_time", 'actions', title='Prediction with DLT')
```

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
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

    2025-02-16 16:40:43 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:40:58 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:41:16 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:41:30 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:41:45 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:41:58 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:42:12 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:42:26 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:42:40 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:42:55 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:43:08 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:43:27 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:43:43 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:43:59 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:44:13 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:44:27 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:44:40 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:44:56 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:45:12 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:45:29 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:45:43 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:45:58 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:46:14 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:46:27 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:46:40 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:46:55 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:47:11 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:47:24 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:47:41 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:47:55 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:48:08 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:48:20 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:48:30 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:48:39 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:48:49 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    /opt/conda/lib/python3.8/site-packages/orbit/template/ets.py:129: RuntimeWarning: Mean of empty slice
      ss[idx] = np.nanmean(adjusted_response[idx :: self._seasonality])
    /opt/conda/lib/python3.8/site-packages/numpy/lib/nanfunctions.py:1878: RuntimeWarning: Degrees of freedom <= 0 for slice.
      var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    2025-02-16 16:48:58 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Input In [116], in <module>
    ----> 1 bt_roll_reg_MCMC_apr.fit_predict()
    

    File /opt/conda/lib/python3.8/site-packages/orbit/diagnostics/backtest.py:383, in BackTester.fit_predict(self)
        381 for train_df, test_df, scheme, key in splitter.split():
        382     model_copy = deepcopy(model)
    --> 383     model_copy.fit(train_df)
        384     train_predictions = model_copy.predict(train_df)
        385     test_predictions = model_copy.predict(test_df)
    

    File /opt/conda/lib/python3.8/site-packages/orbit/forecaster/full_bayes.py:36, in FullBayesianForecaster.fit(self, df, point_method, keep_samples, sampling_temperature, **kwargs)
         28 def fit(
         29     self,
         30     df,
       (...)
         34     **kwargs,
         35 ):
    ---> 36     super().fit(df, sampling_temperature=sampling_temperature, **kwargs)
         37     self._point_method = point_method
         39     if point_method is not None:
    

    File /opt/conda/lib/python3.8/site-packages/orbit/forecaster/forecaster.py:162, in Forecaster.fit(self, df, **kwargs)
        158 model_param_names = self._model.get_model_param_names()
        160 # note that estimator will search for the .stan, .pyro model file based on the
        161 # estimator type and model_name provided
    --> 162 _posterior_samples, training_metrics = estimator.fit(
        163     model_name=model_name,
        164     model_param_names=model_param_names,
        165     data_input=data_input,
        166     fitter=self._model.get_fitter(),
        167     init_values=init_values,
        168     **kwargs,
        169 )
        170 self._posterior_samples = _posterior_samples
        171 self._training_metrics = training_metrics
    

    File /opt/conda/lib/python3.8/site-packages/orbit/estimators/stan_estimator.py:140, in StanEstimatorMCMC.fit(self, model_name, model_param_names, sampling_temperature, data_input, fitter, init_values)
        137 compiled_mod = get_compiled_stan_model(model_name)
        138 # check https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.sample
        139 # for additional args
    --> 140 stan_mcmc_fit = compiled_mod.sample(
        141     data=data_input,
        142     iter_sampling=self._num_sample_per_chain,
        143     iter_warmup=self._num_warmup_per_chain,
        144     chains=self.chains,
        145     parallel_chains=self.cores,
        146     inits=init_values,
        147     seed=self.seed,
        148     **self._stan_mcmc_args,
        149 )
        151 stan_extract = stan_mcmc_fit.stan_variables()
        152 posteriors = {
        153     param: stan_extract[param] for param in model_param_names + ["loglk"]
        154 }
    

    File /opt/conda/lib/python3.8/site-packages/cmdstanpy/model.py:1201, in CmdStanModel.sample(self, data, chains, parallel_chains, threads_per_chain, seed, chain_ids, inits, iter_warmup, iter_sampling, save_warmup, thin, max_treedepth, metric, step_size, adapt_engaged, adapt_delta, adapt_init_phase, adapt_metric_window, adapt_step_size, fixed_param, output_dir, sig_figs, save_latent_dynamics, save_profile, show_progress, show_console, refresh, time_fmt, timeout, force_one_process_per_chain)
       1194 if not runset._check_retcodes():
       1195     msg = (
       1196         f'Error during sampling:\n{errors}\n'
       1197         + f'Command and output files:\n{repr(runset)}\n'
       1198         + 'Consider re-running with show_console=True if the above'
       1199         + ' output is unclear!'
       1200     )
    -> 1201     raise RuntimeError(msg)
       1202 if errors:
       1203     msg = (
       1204         f'Non-fatal error during sampling:\n{errors}\n'
       1205         + 'Consider re-running with show_console=True if the above'
       1206         + ' output is unclear!'
       1207     )
    

    RuntimeError: Error during sampling:
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Command and output files:
    RunSet: chains=4, chain_ids=[1, 2, 3, 4], num_processes=4
     cmd (chain 1):
    	['/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt', 'id=1', 'random', 'seed=8888', 'data', 'file=/tmp/tmpqqnjd0l9/q61yx6f3.json', 'init=/tmp/tmpqqnjd0l9/8e8xa8cw.json', 'output', 'file=/tmp/tmpqqnjd0l9/dlts1b58dem/dlt-20250216164858_1.csv', 'method=sample', 'num_samples=25', 'num_warmup=225', 'algorithm=hmc', 'adapt', 'engaged=1']
     retcodes=[1, 1, 1, 1]
     per-chain output files (showing chain 1 only):
     csv_file:
    	/tmp/tmpqqnjd0l9/dlts1b58dem/dlt-20250216164858_1.csv
     console_msgs (if any):
    	/tmp/tmpqqnjd0l9/dlts1b58dem/dlt-20250216164858_0-stdout.txt
    Consider re-running with show_console=True if the above output is unclear!



```python
params_MCMC_2 = reg_MCMC_model_apr.get_posterior_samples(permute=False) #достаём информацию о параметрах
az.plot_trace(params_MCMC_2, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```




    array([[<Axes: title={'center': 'obs_sigma'}>,
            <Axes: title={'center': 'obs_sigma'}>]], dtype=object)




    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_96_1.png)
    



```python
bt_roll_reg_MCMC_apr.score() #выводим метрики
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

    2025-02-16 16:52:57 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    




    <orbit.forecaster.full_bayes.FullBayesianForecaster at 0x7f62324797c0>




```python
predicted_df_reg_MCMC = reg_MCMC_model_rl.predict(df=test_dat)
_ = plot_predicted_data(mcmc_date, predicted_df_reg_MCMC, "hour_time", 'actions', title='Prediction with DLT')
```

    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial
    


    
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

    2025-02-16 16:56:52 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:57:08 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:57:23 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:57:38 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:57:56 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:58:10 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:58:24 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:58:39 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:58:53 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:59:09 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:59:25 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 16:59:41 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:00:01 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:00:17 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:00:32 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:00:46 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:01:04 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:01:19 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:01:35 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:01:50 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:02:07 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:02:20 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:02:33 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:02:47 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:03:04 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:03:17 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:03:32 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:03:46 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:04:01 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:04:23 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:04:38 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:04:52 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:05:06 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:05:17 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    2025-02-16 17:05:27 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    

    /opt/conda/lib/python3.8/site-packages/orbit/template/ets.py:129: RuntimeWarning: Mean of empty slice
      ss[idx] = np.nanmean(adjusted_response[idx :: self._seasonality])
    /opt/conda/lib/python3.8/site-packages/numpy/lib/nanfunctions.py:1878: RuntimeWarning: Degrees of freedom <= 0 for slice.
      var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    2025-02-16 17:05:36 - orbit - INFO - Sampling (PyStan) with chains: 4, cores: 8, temperature: 1.000, warmups (per chain): 225 and samples(per chain): 25.
    


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Input In [124], in <module>
    ----> 1 bt_roll_reg_MCMC.fit_predict()
    

    File /opt/conda/lib/python3.8/site-packages/orbit/diagnostics/backtest.py:383, in BackTester.fit_predict(self)
        381 for train_df, test_df, scheme, key in splitter.split():
        382     model_copy = deepcopy(model)
    --> 383     model_copy.fit(train_df)
        384     train_predictions = model_copy.predict(train_df)
        385     test_predictions = model_copy.predict(test_df)
    

    File /opt/conda/lib/python3.8/site-packages/orbit/forecaster/full_bayes.py:36, in FullBayesianForecaster.fit(self, df, point_method, keep_samples, sampling_temperature, **kwargs)
         28 def fit(
         29     self,
         30     df,
       (...)
         34     **kwargs,
         35 ):
    ---> 36     super().fit(df, sampling_temperature=sampling_temperature, **kwargs)
         37     self._point_method = point_method
         39     if point_method is not None:
    

    File /opt/conda/lib/python3.8/site-packages/orbit/forecaster/forecaster.py:162, in Forecaster.fit(self, df, **kwargs)
        158 model_param_names = self._model.get_model_param_names()
        160 # note that estimator will search for the .stan, .pyro model file based on the
        161 # estimator type and model_name provided
    --> 162 _posterior_samples, training_metrics = estimator.fit(
        163     model_name=model_name,
        164     model_param_names=model_param_names,
        165     data_input=data_input,
        166     fitter=self._model.get_fitter(),
        167     init_values=init_values,
        168     **kwargs,
        169 )
        170 self._posterior_samples = _posterior_samples
        171 self._training_metrics = training_metrics
    

    File /opt/conda/lib/python3.8/site-packages/orbit/estimators/stan_estimator.py:140, in StanEstimatorMCMC.fit(self, model_name, model_param_names, sampling_temperature, data_input, fitter, init_values)
        137 compiled_mod = get_compiled_stan_model(model_name)
        138 # check https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.sample
        139 # for additional args
    --> 140 stan_mcmc_fit = compiled_mod.sample(
        141     data=data_input,
        142     iter_sampling=self._num_sample_per_chain,
        143     iter_warmup=self._num_warmup_per_chain,
        144     chains=self.chains,
        145     parallel_chains=self.cores,
        146     inits=init_values,
        147     seed=self.seed,
        148     **self._stan_mcmc_args,
        149 )
        151 stan_extract = stan_mcmc_fit.stan_variables()
        152 posteriors = {
        153     param: stan_extract[param] for param in model_param_names + ["loglk"]
        154 }
    

    File /opt/conda/lib/python3.8/site-packages/cmdstanpy/model.py:1201, in CmdStanModel.sample(self, data, chains, parallel_chains, threads_per_chain, seed, chain_ids, inits, iter_warmup, iter_sampling, save_warmup, thin, max_treedepth, metric, step_size, adapt_engaged, adapt_delta, adapt_init_phase, adapt_metric_window, adapt_step_size, fixed_param, output_dir, sig_figs, save_latent_dynamics, save_profile, show_progress, show_console, refresh, time_fmt, timeout, force_one_process_per_chain)
       1194 if not runset._check_retcodes():
       1195     msg = (
       1196         f'Error during sampling:\n{errors}\n'
       1197         + f'Command and output files:\n{repr(runset)}\n'
       1198         + 'Consider re-running with show_console=True if the above'
       1199         + ' output is unclear!'
       1200     )
    -> 1201     raise RuntimeError(msg)
       1202 if errors:
       1203     msg = (
       1204         f'Non-fatal error during sampling:\n{errors}\n'
       1205         + 'Consider re-running with show_console=True if the above'
       1206         + ' output is unclear!'
       1207     )
    

    RuntimeError: Error during sampling:
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Exception: dlt_model_namespace::dlt_model: SEASONALITY_SD is nan, but must be greater than or equal to 0.000000 (in '/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt.stan', line 74, column 2 to column 31)
    Command and output files:
    RunSet: chains=4, chain_ids=[1, 2, 3, 4], num_processes=4
     cmd (chain 1):
    	['/opt/conda/lib/python3.8/site-packages/orbit/stan/dlt', 'id=1', 'random', 'seed=8888', 'data', 'file=/tmp/tmpqqnjd0l9/ngllxkyc.json', 'init=/tmp/tmpqqnjd0l9/f4nhwtkf.json', 'output', 'file=/tmp/tmpqqnjd0l9/dltfbzf3aay/dlt-20250216170536_1.csv', 'method=sample', 'num_samples=25', 'num_warmup=225', 'algorithm=hmc', 'adapt', 'engaged=1']
     retcodes=[1, 1, 1, 1]
     per-chain output files (showing chain 1 only):
     csv_file:
    	/tmp/tmpqqnjd0l9/dltfbzf3aay/dlt-20250216170536_1.csv
     console_msgs (if any):
    	/tmp/tmpqqnjd0l9/dltfbzf3aay/dlt-20250216170536_0-stdout.txt
    Consider re-running with show_console=True if the above output is unclear!



```python
params_MCMC_3 = reg_MCMC_model_rl.get_posterior_samples(permute=False) #достаём информацию о параметрах
az.plot_trace(params_MCMC_3, chain_prop={"color": ['r', 'b', 'g', 'y']}, var_names = "obs_sigma")
```




    array([[<Axes: title={'center': 'obs_sigma'}>,
            <Axes: title={'center': 'obs_sigma'}>]], dtype=object)




    
![png](Lesson%205.2_predict_metric%20_files/Lesson%205.2_predict_metric%20_103_1.png)
    



```python
bt_roll_reg_MCMC.score()
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
