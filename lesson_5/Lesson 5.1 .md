# Библиотеки


```python
pip install pandahouse
```

    Requirement already satisfied: pandahouse in /opt/conda/lib/python3.8/site-packages (0.2.7)
    Requirement already satisfied: toolz in /opt/conda/lib/python3.8/site-packages (from pandahouse) (1.0.0)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.8/site-packages (from pandahouse) (1.4.2)
    Requirement already satisfied: requests in /opt/conda/lib/python3.8/site-packages (from pandahouse) (2.27.1)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.8/site-packages (from pandas->pandahouse) (2021.3)
    Requirement already satisfied: python-dateutil>=2.8.1 in /opt/conda/lib/python3.8/site-packages (from pandas->pandahouse) (2.8.2)
    Requirement already satisfied: numpy>=1.18.5 in /opt/conda/lib/python3.8/site-packages (from pandas->pandahouse) (1.22.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (1.26.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (2.0.11)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.8/site-packages (from requests->pandahouse) (3.3)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.8/site-packages (from python-dateutil>=2.8.1->pandas->pandahouse) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install tensorflow
```

    Requirement already satisfied: tensorflow in /opt/conda/lib/python3.8/site-packages (2.13.1)
    Requirement already satisfied: typing-extensions<4.6.0,>=3.6.6 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (4.0.1)
    Requirement already satisfied: tensorboard<2.14,>=2.13 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (2.13.0)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (4.25.6)
    Requirement already satisfied: h5py>=2.9.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (3.11.0)
    Requirement already satisfied: tensorflow-estimator<2.14,>=2.13.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (2.13.0)
    Requirement already satisfied: six>=1.12.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (2.4.0)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.8/site-packages (from tensorflow) (60.8.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (3.4.0)
    Requirement already satisfied: flatbuffers>=23.1.21 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (25.2.10)
    Requirement already satisfied: astunparse>=1.6.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: google-pasta>=0.1.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: numpy<=1.24.3,>=1.22 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (1.22.4)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (0.4.0)
    Requirement already satisfied: keras<2.14,>=2.13.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (2.13.1)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (1.70.0)
    Requirement already satisfied: absl-py>=1.0.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (2.1.0)
    Requirement already satisfied: wrapt>=1.11.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (1.17.2)
    Requirement already satisfied: libclang>=13.0.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (18.1.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow) (0.34.0)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.8/site-packages (from tensorflow) (21.3)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.8/site-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow) (2.38.0)
    Requirement already satisfied: werkzeug>=1.0.1 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow) (3.0.6)
    Requirement already satisfied: markdown>=2.6.8 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow) (3.7)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow) (0.7.2)
    Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow) (1.0.0)
    Requirement already satisfied: requests<3,>=2.21.0 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow) (2.27.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.8/site-packages (from packaging->tensorflow) (3.0.7)
    Requirement already satisfied: rsa<5,>=3.1.4 in /opt/conda/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow) (4.9)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow) (0.4.1)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/conda/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow) (5.5.1)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /opt/conda/lib/python3.8/site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow) (2.0.0)
    Requirement already satisfied: importlib-metadata>=4.4 in /opt/conda/lib/python3.8/site-packages (from markdown>=2.6.8->tensorboard<2.14,>=2.13->tensorflow) (4.11.0)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow) (1.26.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow) (2.0.11)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow) (2021.10.8)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /opt/conda/lib/python3.8/site-packages (from werkzeug>=1.0.1->tensorboard<2.14,>=2.13->tensorflow) (2.1.5)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/lib/python3.8/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.14,>=2.13->tensorflow) (3.7.0)
    Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /opt/conda/lib/python3.8/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow) (0.6.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /opt/conda/lib/python3.8/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow) (3.2.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install tfcausalimpact
```

    Requirement already satisfied: tfcausalimpact in /opt/conda/lib/python3.8/site-packages (0.0.18)
    Requirement already satisfied: tensorflow-probability[tf]<=0.25,>=0.18 in /opt/conda/lib/python3.8/site-packages (from tfcausalimpact) (0.21.0)
    Requirement already satisfied: pandas<=2.2,>=1.3.5 in /opt/conda/lib/python3.8/site-packages (from tfcausalimpact) (1.4.2)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.8/site-packages (from tfcausalimpact) (3.0.3)
    Requirement already satisfied: tensorflow>=2.10 in /opt/conda/lib/python3.8/site-packages (from tfcausalimpact) (2.13.1)
    Requirement already satisfied: matplotlib in /opt/conda/lib/python3.8/site-packages (from tfcausalimpact) (3.4.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /opt/conda/lib/python3.8/site-packages (from pandas<=2.2,>=1.3.5->tfcausalimpact) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.8/site-packages (from pandas<=2.2,>=1.3.5->tfcausalimpact) (2021.3)
    Requirement already satisfied: numpy>=1.18.5 in /opt/conda/lib/python3.8/site-packages (from pandas<=2.2,>=1.3.5->tfcausalimpact) (1.22.4)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (0.4.0)
    Requirement already satisfied: astunparse>=1.6.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (1.6.3)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (0.34.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (1.70.0)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (60.8.1)
    Requirement already satisfied: absl-py>=1.0.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (2.1.0)
    Requirement already satisfied: wrapt>=1.11.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (1.17.2)
    Requirement already satisfied: keras<2.14,>=2.13.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (2.13.1)
    Requirement already satisfied: tensorflow-estimator<2.14,>=2.13.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (2.13.0)
    Requirement already satisfied: tensorboard<2.14,>=2.13 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (2.13.0)
    Requirement already satisfied: flatbuffers>=23.1.21 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (25.2.10)
    Requirement already satisfied: libclang>=13.0.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (18.1.1)
    Requirement already satisfied: termcolor>=1.1.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (2.4.0)
    Requirement already satisfied: typing-extensions<4.6.0,>=3.6.6 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (4.0.1)
    Requirement already satisfied: h5py>=2.9.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (3.11.0)
    Requirement already satisfied: google-pasta>=0.1.1 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (0.2.0)
    Requirement already satisfied: six>=1.12.0 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (1.16.0)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (21.3)
    Requirement already satisfied: opt-einsum>=2.3.2 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (3.4.0)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /opt/conda/lib/python3.8/site-packages (from tensorflow>=2.10->tfcausalimpact) (4.25.6)
    [33mWARNING: tensorflow-probability 0.21.0 does not provide the extra 'tf'[0m[33m
    [0mRequirement already satisfied: cloudpickle>=1.3 in /opt/conda/lib/python3.8/site-packages (from tensorflow-probability[tf]<=0.25,>=0.18->tfcausalimpact) (3.1.1)
    Requirement already satisfied: dm-tree in /opt/conda/lib/python3.8/site-packages (from tensorflow-probability[tf]<=0.25,>=0.18->tfcausalimpact) (0.1.8)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.8/site-packages (from tensorflow-probability[tf]<=0.25,>=0.18->tfcausalimpact) (5.1.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.8/site-packages (from jinja2->tfcausalimpact) (2.1.5)
    Requirement already satisfied: pyparsing>=2.2.1 in /opt/conda/lib/python3.8/site-packages (from matplotlib->tfcausalimpact) (3.0.7)
    Requirement already satisfied: pillow>=6.2.0 in /opt/conda/lib/python3.8/site-packages (from matplotlib->tfcausalimpact) (9.5.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.8/site-packages (from matplotlib->tfcausalimpact) (1.4.4)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.8/site-packages (from matplotlib->tfcausalimpact) (0.11.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.8/site-packages (from astunparse>=1.6.0->tensorflow>=2.10->tfcausalimpact) (0.37.1)
    Requirement already satisfied: werkzeug>=1.0.1 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (3.0.6)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (2.38.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (0.7.2)
    Requirement already satisfied: markdown>=2.6.8 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (3.7)
    Requirement already satisfied: requests<3,>=2.21.0 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (2.27.1)
    Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in /opt/conda/lib/python3.8/site-packages (from tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (1.0.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /opt/conda/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (4.9)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/conda/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (5.5.1)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (0.4.1)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /opt/conda/lib/python3.8/site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (2.0.0)
    Requirement already satisfied: importlib-metadata>=4.4 in /opt/conda/lib/python3.8/site-packages (from markdown>=2.6.8->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (4.11.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (1.26.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (2.0.11)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (3.3)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/lib/python3.8/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (3.7.0)
    Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /opt/conda/lib/python3.8/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (0.6.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /opt/conda/lib/python3.8/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow>=2.10->tfcausalimpact) (3.2.0)
    Note: you may need to restart the kernel to use updated packages.
    


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
