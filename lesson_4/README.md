<details>
  <summary> Задание 1 </summary>


К нам пришли наши коллеги из ML-отдела и рассказали, что планируют выкатывать новый алгоритм, рекомендующий нашим пользователям интересные посты. После обсуждений того, как он это делает, вы пришли к следующему пониманию:

Алгоритм добавляет пользователям 1-2 просмотра
Вероятность того, что он сработает, составляет 90%
Если у пользователя меньше 50 просмотров, то алгоритм не сработает
Вы предполагаете, что увеличение числа просмотров приведёт и к увеличению лайков на пользователя. Встаёт вопрос: сможем ли мы обнаружить различия в среднем количестве лайков на пользователя? Чтобы ответить на этот вопрос, давайте проведём симуляцию Монте-Карло.

#### Выгрузим данные 


```python
import pandas as pd
import pandahouse as ph
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import norm
from scipy.stats import ttest_ind
import numpy as np
from tqdm import tqdm
```


```python
connection = {'host': 'https://clickhouse.lab.karpov.courses',
'database':'simulator_20250120',
'user':'student',
'password':'dpo_python_2020'}
```


```python
q = """
select views, count() as users
from (select user_id,
             sum(action = 'view') as views
from {db}.feed_actions 
where toDate(time) BETWEEN '2024-12-27' AND  '2025-01-02'

   
group by user_id
)
group by views
order by views
"""
```


```python
views_distribution = ph.read_clickhouse(q, connection=connection)
views_distribution['p'] = views_distribution['users']/views_distribution.users.sum()
views_distribution.sort_values(by = 'p', ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>views</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>545</td>
      <td>0.012977</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>537</td>
      <td>0.012787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>500</td>
      <td>0.011906</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>485</td>
      <td>0.011548</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>469</td>
      <td>0.011167</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>280</th>
      <td>287</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>278</th>
      <td>285</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>276</th>
      <td>280</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>300</th>
      <td>370</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
  </tbody>
</table>
<p>301 rows × 3 columns</p>
</div>




```python
views_distr = stats.rv_discrete(name='views_distr', 
                                values=(views_distribution['views'], 
                                        views_distribution['p']))
```


```python
q_1 = """
select 
   floor(ctr, 2) as ctr, count() as users
from (select toDate(time) as dt, 
    user_id,
    sum(action = 'like')/sum(action = 'view') as ctr
from {db}.feed_actions
where dt BETWEEN '2024-12-27' AND  '2025-01-02'

group by dt, user_id
)
group by ctr
"""
```


```python
ctr_distribution = ph.read_clickhouse(q_1, connection=connection)
ctr_distribution['p'] = ctr_distribution['users']/ctr_distribution.users.sum()
ctr_distribution.sort_values(by = 'p', ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ctr</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>0.20</td>
      <td>4993</td>
      <td>0.058658</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.16</td>
      <td>4233</td>
      <td>0.049729</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.25</td>
      <td>4216</td>
      <td>0.049529</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.18</td>
      <td>4213</td>
      <td>0.049494</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.21</td>
      <td>3957</td>
      <td>0.046487</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.81</td>
      <td>2</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.83</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.00</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.73</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.88</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 3 columns</p>
</div>




```python
views = views_distribution['views']
```


```python
views_prob = views_distribution['users'] / views_distribution['users'].sum()
```


```python
ctrs = ctr_distribution['ctr']
```


```python
ctrs_prob = ctr_distribution['users'] / ctr_distribution['users'].sum()
```


```python
rng = np.random.default_rng()
```


```python
#n_users = 30591
n_users = views_distribution.users.sum() // 2
n_users


```




    20998




```python
pvalues = []
for _ in tqdm(range(10000)):
    group_A_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views += ((1 + rng.binomial(n=1, p=0.5, size=n_users)) * rng.binomial(n=1, p=0.9, size=n_users) * (group_B_views >= 50))
    group_A_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    group_B_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    group_A_likes = rng.binomial(n=group_A_views, p=group_A_ctrs)
    group_B_likes = rng.binomial(n=group_B_views, p=group_B_ctrs)
   

    _, p_value = ttest_ind(group_A_likes, group_B_likes, equal_var=False)
    pvalues.append(p_value)

# Вычисление доли значимых различий
alpha = 0.05
print(np.mean(np.array(pvalues) < alpha) * 100)

```

    100%|████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:25<00:00, 48.74it/s]

    25.95
    

    
    

Ответ: 25.9

</details>

<details>
  <summary> Задание 2 </summary>


К нам снова пришли коллеги из ML-отдела с радостной новостью: они улучшили качество алгоритма! Теперь он срабатывает на пользователях с числом просмотров от 30 и выше.

#### Выгрузим данные 


```python
import pandas as pd
import pandahouse as ph
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import norm
from scipy.stats import ttest_ind
import numpy as np
from tqdm import tqdm
```


```python
connection = {'host': 'https://clickhouse.lab.karpov.courses',
'database':'simulator_20250120',
'user':'student',
'password':'dpo_python_2020'}
```


```python
q = """
select views, count() as users
from (select user_id,
             sum(action = 'view') as views
from {db}.feed_actions 
where toDate(time) BETWEEN '2024-12-27' AND  '2025-01-02'

   
group by user_id
)
group by views
order by views
"""
```


```python
views_distribution = ph.read_clickhouse(q, connection=connection)
views_distribution['p'] = views_distribution['users']/views_distribution.users.sum()
views_distribution.sort_values(by = 'p', ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>views</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>545</td>
      <td>0.012977</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>537</td>
      <td>0.012787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>500</td>
      <td>0.011906</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>485</td>
      <td>0.011548</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>469</td>
      <td>0.011167</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>280</th>
      <td>287</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>278</th>
      <td>285</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>276</th>
      <td>280</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>300</th>
      <td>370</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
  </tbody>
</table>
<p>301 rows × 3 columns</p>
</div>




```python
views_distr = stats.rv_discrete(name='views_distr', 
                                values=(views_distribution['views'], 
                                        views_distribution['p']))
```


```python
q_1 = """
select 
   floor(ctr, 2) as ctr, count() as users
from (select toDate(time) as dt, 
    user_id,
    sum(action = 'like')/sum(action = 'view') as ctr
from {db}.feed_actions
where dt BETWEEN '2024-12-27' AND  '2025-01-02'

group by dt, user_id
)
group by ctr
"""
```


```python
ctr_distribution = ph.read_clickhouse(q_1, connection=connection)
ctr_distribution['p'] = ctr_distribution['users']/ctr_distribution.users.sum()
ctr_distribution.sort_values(by = 'p', ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ctr</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>0.20</td>
      <td>4993</td>
      <td>0.058658</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.16</td>
      <td>4233</td>
      <td>0.049729</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.25</td>
      <td>4216</td>
      <td>0.049529</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.18</td>
      <td>4213</td>
      <td>0.049494</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.21</td>
      <td>3957</td>
      <td>0.046487</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.81</td>
      <td>2</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.83</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.00</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.73</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.88</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 3 columns</p>
</div>




```python
views = views_distribution['views']
```


```python
views_prob = views_distribution['users'] / views_distribution['users'].sum()
```


```python
ctrs = ctr_distribution['ctr']
```


```python
ctrs_prob = ctr_distribution['users'] / ctr_distribution['users'].sum()
```


```python
rng = np.random.default_rng()
```


```python
n_users = views_distribution.users.sum() // 2
```




    20998




```python
pvalues = []
for _ in tqdm(range(20000)):
    group_A_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views += ((1 + rng.binomial(n=1, p=0.5, size=n_users)) * rng.binomial(n=1, p=0.9, size=n_users) * (group_B_views >= 30))
    group_A_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    group_B_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    group_A_likes = rng.binomial(n=group_A_views, p=group_A_ctrs)
    group_B_likes = rng.binomial(n=group_B_views, p=group_B_ctrs)
   

    _, p_value = ttest_ind(group_A_likes, group_B_likes, equal_var=False)
    pvalues.append(p_value)

# Вычисление доли значимых различий
alpha = 0.05
print(np.mean(np.array(pvalues) < alpha) * 100)

```

    100%|████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:38<00:00, 45.82it/s]

    41.67
   

Ответ: 42.0

</details>

<details>
  <summary> Задание 3 </summary>

Теперь нас пришло радовать начальство: нам утвердили длительность эксперимента длиной в 2 недели! Давайте теперь допустим, что в эти две недели к нам придёт столько же пользователей, сколько пришло суммарно за период АА-теста и АБ-теста (опять же, смотрите диапазон дат в прошлом уроке).

#### Выгрузим данные 


```python
import pandas as pd
import pandahouse as ph
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import norm
from scipy.stats import ttest_ind
import numpy as np
from tqdm import tqdm
```


```python
connection = {'host': 'https://clickhouse.lab.karpov.courses',
'database':'simulator_20250120',
'user':'student',
'password':'dpo_python_2020'}
```


```python
q = """
select views, count() as users
from (select user_id,
             sum(action = 'view') as views
from {db}.feed_actions 
where toDate(time) BETWEEN '2024-12-27' AND  '2025-01-02'

   
group by user_id
)
group by views
order by views
"""
```


```python
views_distribution = ph.read_clickhouse(q, connection=connection)
views_distribution['p'] = views_distribution['users']/views_distribution.users.sum()
views_distribution.sort_values(by = 'p', ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>views</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>545</td>
      <td>0.012977</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>537</td>
      <td>0.012787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>500</td>
      <td>0.011906</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>485</td>
      <td>0.011548</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>469</td>
      <td>0.011167</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>280</th>
      <td>287</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>278</th>
      <td>285</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>276</th>
      <td>280</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>300</th>
      <td>370</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
  </tbody>
</table>
<p>301 rows × 3 columns</p>
</div>




```python
views_distr = stats.rv_discrete(name='views_distr', 
                                values=(views_distribution['views'], 
                                        views_distribution['p']))
```


```python
q_1 = """
select 
   floor(ctr, 2) as ctr, count() as users
from (select toDate(time) as dt, 
    user_id,
    sum(action = 'like')/sum(action = 'view') as ctr
from {db}.feed_actions
where dt BETWEEN '2024-12-27' AND  '2025-01-02'

group by dt, user_id
)
group by ctr
"""
```


```python
ctr_distribution = ph.read_clickhouse(q_1, connection=connection)
ctr_distribution['p'] = ctr_distribution['users']/ctr_distribution.users.sum()
ctr_distribution.sort_values(by = 'p', ascending = False)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ctr</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>0.20</td>
      <td>4993</td>
      <td>0.058658</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.16</td>
      <td>4233</td>
      <td>0.049729</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.25</td>
      <td>4216</td>
      <td>0.049529</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.18</td>
      <td>4213</td>
      <td>0.049494</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.21</td>
      <td>3957</td>
      <td>0.046487</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.81</td>
      <td>2</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.83</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.00</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.73</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.88</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 3 columns</p>
</div>




```python
views = views_distribution['views']
```


```python
views_prob = views_distribution['users'] / views_distribution['users'].sum()
```


```python
ctrs = ctr_distribution['ctr']
```


```python
ctrs_prob = ctr_distribution['users'] / ctr_distribution['users'].sum()
```


```python
rng = np.random.default_rng()
```


```python
n_users = 30591
```


```python
pvalues = []
for _ in tqdm(range(20000)):
    group_A_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views += ((1 + rng.binomial(n=1, p=0.5, size=n_users)) * rng.binomial(n=1, p=0.9, size=n_users) * (group_B_views >= 30))
    group_A_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    group_B_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    group_A_likes = rng.binomial(n=group_A_views, p=group_A_ctrs)
    group_B_likes = rng.binomial(n=group_B_views, p=group_B_ctrs)
   

    _, p_value = ttest_ind(group_A_likes, group_B_likes, equal_var=False)
    pvalues.append(p_value)

# Вычисление доли значимых различий
alpha = 0.05
print(np.mean(np.array(pvalues) < alpha) * 100)

```

    100%|████████████████████████████████████████████████████████████████████████████| 20000/20000 [10:19<00:00, 32.28it/s]

    56.08
    

Ответ: 56.0

</details>

<details>
  <summary> Задание 4 </summary>
  
Всё это время мы анализировали наши выборки целиком — и тех пользователей, на которых алгоритм повлиял, и тех, кого он не мог затронуть (меньше 30 просмотров). А что, если мы будем отбирать только нужных пользователей и скармливать t-тесту именно их? Да, выборка будет меньше, но мы избавимся от мусора — а значит, и чувствительность наверняка будет выше. В ответе укажите получившуюся мощность.

#### Выгрузим данные 

```python
import pandas as pd
import pandahouse as ph
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import norm
from scipy.stats import ttest_ind
import numpy as np
from tqdm import tqdm


```


```python
connection = {'host': 'https://clickhouse.lab.karpov.courses',
'database':'simulator_20250120',
'user':'student',
'password':'dpo_python_2020'}
```


```python
q = """
select views, count() as users
from (select user_id,
             sum(action = 'view') as views
from {db}.feed_actions 
where toDate(time) BETWEEN '2024-12-27' AND  '2025-01-02'

   
group by user_id
)
group by views
order by views
"""
```


```python
views_distribution = ph.read_clickhouse(q, connection=connection)
views_distribution['p'] = views_distribution['users']/views_distribution.users.sum()
views_distribution.sort_values(by = 'p', ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>views</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>545</td>
      <td>0.012977</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>537</td>
      <td>0.012787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>500</td>
      <td>0.011906</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>485</td>
      <td>0.011548</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>469</td>
      <td>0.011167</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>280</th>
      <td>287</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>278</th>
      <td>285</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>276</th>
      <td>280</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>300</th>
      <td>370</td>
      <td>1</td>
      <td>0.000024</td>
    </tr>
  </tbody>
</table>
<p>301 rows × 3 columns</p>
</div>




```python
views_distr = stats.rv_discrete(name='views_distr', 
                                values=(views_distribution['views'], 
                                        views_distribution['p']))
```


```python
q_1 = """
select 
   floor(ctr, 2) as ctr, count() as users
from (select toDate(time) as dt, 
    user_id,
    sum(action = 'like')/sum(action = 'view') as ctr
from {db}.feed_actions
where dt BETWEEN '2024-12-27' AND  '2025-01-02'

group by dt, user_id
)
group by ctr
"""
```


```python
ctr_distribution = ph.read_clickhouse(q_1, connection=connection)
ctr_distribution['p'] = ctr_distribution['users']/ctr_distribution.users.sum()
ctr_distribution.sort_values(by = 'p', ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ctr</th>
      <th>users</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>0.20</td>
      <td>4993</td>
      <td>0.058658</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.16</td>
      <td>4233</td>
      <td>0.049729</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.25</td>
      <td>4216</td>
      <td>0.049529</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.18</td>
      <td>4213</td>
      <td>0.049494</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.21</td>
      <td>3957</td>
      <td>0.046487</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.81</td>
      <td>2</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.83</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.00</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.73</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.88</td>
      <td>1</td>
      <td>0.000012</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 3 columns</p>
</div>




```python
views = views_distribution['views']
```


```python
views_prob = views_distribution['users'] / views_distribution['users'].sum()
```


```python
ctrs = ctr_distribution['ctr']
```


```python
ctrs_prob = ctr_distribution['users'] / ctr_distribution['users'].sum()
```


```python
rng = np.random.default_rng()
```


```python
n_users = 30591
```


```python
pvalues = []
for _ in tqdm(range(20000)):
    group_A_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views = rng.choice(views_distribution['views'], size=n_users, replace=True, p=views_distribution['p']).astype(np.int64)
    group_B_views += ((1 + rng.binomial(n=1, p=0.5, size=n_users)) * rng.binomial(n=1, p=0.9, size=n_users) * (group_B_views >= 30))
    group_A_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    group_B_ctrs = rng.choice(ctrs, size=n_users, replace=True, p=ctrs_prob)
    mask_A = group_A_views >= 30
    mask_B = group_B_views >= 30
    group_A_likes = rng.binomial(n=group_A_views, p=group_A_ctrs)
    group_B_likes = rng.binomial(n=group_B_views, p=group_B_ctrs)
   

    _, p_value = ttest_ind(group_A_likes[mask_A], group_B_likes[mask_B], equal_var=False)
    pvalues.append(p_value)

# Вычисление доли значимых различий
alpha = 0.05
print(np.mean(np.array(pvalues) < alpha) * 100)

```

    100%|████████████████████████████████████████████████████████████████████████████| 20000/20000 [09:10<00:00, 36.31it/s]

    64.495
    

    
    


```python
Ответ: 65.19
```


```python

```


```python

```


</details>
