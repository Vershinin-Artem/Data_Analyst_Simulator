# Задание 1
В лекции мы выяснили, что при проведении А/А-теста было бы здорово убедиться в том, что наша система сплитования работает корректно, и ключевая метрика не отличается между группами не только в конкретно нашем А/А-тесте, но и в целом.

В идеале было бы здорово провести бесконечное количество А/A-тестов и посмотреть, в каком количестве случаев нам удалось отклонить нулевую гипотезу. Если система сплитования работает корректно, то статистически значимые различия между двумя группами встречались бы только в результате случайного ложного срабатывания. Например, если мы отвергаем нулевую гипотезу при условии, что p_value < 0.05, то только приблизительно в 5% случаев у нас бы получались статистические значимые различия между 0 и 1 группой.

Понятное дело, что на практике провести бесконечное число тестов у нас вряд ли получится, поэтому используется небольшой трюк. Мы будем многократно извлекать подвыборки из наших данных, проводить t-test, а в конце посмотрим, в каком проценте случаев нам удалось отклонить нулевую гипотезу.

Сделаем следующее:

Берём данные АА-теста из следующего диапазона: с '2024-12-27' по '2025-01-02.
Из групп 2 и 3 берём подвыборки без возвращения размером в 500 юзеров.
Сравниваем их t-тестом и сохраняем p-value (здесь и далее используем аргумент equal_var=False).
Повторяем это 10000 раз.
Нарисуйте гистограмму получившихся p-value и посчитайте долю p-value, оказавшихся ниже порога значимости в 0.05. Что мы можем сказать по этому результату?

``` Python
q = """
SELECT exp_group, 
    user_id,
    sum(action = 'like') as likes,
    sum(action = 'view') as views,
    likes/views as ctr
FROM {db}.feed_actions 
WHERE toDate(time) between '2024-12-27' and '2025-01-02'
    and exp_group in (2,3)
GROUP BY exp_group, user_id
"""

df = ph.read_clickhouse(q, connection=connection)
df.groupby('exp_group').count()
p_values = []
# Повторяем 10 000 раз
for _ in range(10000):
    # Выбираем случайные выборки из каждой группы
    exp_group_2 = df[df.exp_group == 2].ctr.sample(n=500, replace=False, random_state=None)
    exp_group_3 = df[df.exp_group == 3].ctr.sample(n=500, replace=False, random_state=None)
    
    # Выполняем t-тест
    t_statistic, p_value = stats.ttest_ind(exp_group_2, exp_group_3, equal_var=False)
    
    # Сохраняем p-значение
    p_values.append(p_value)

# Преобразуем список p-значений в массив NumPy для удобства

p_values = np.array(p_values)

# Выводим среднее p-значение

print(f"Среднее p-значение: {p_values.mean()}")

Среднее p-значение: 0.503257606618666

plt.hist(p_values, bins=50, edgecolor='black')
plt.title('Распределение p-значений')
plt.xlabel('p-value')
plt.ylabel('Частота')
plt.show()
```

<div style="text-align: center;">
<img src="https://sun9-60.userapi.com/impg/nPmu5hpCKPnFMG8tVmO1cmnd57kzt2UAIebTjw/tRkldY4TIhg.jpg?size=405x305&quality=95&sign=bd9ebfa6b390956d645d69ae03b926ad&type=album" alt="p значение" style="display: inline-block;">
</div>

Распределение получившихся p-value является примерно **равномерным** . Доля p-value ниже порога значимости составляет около **0.046** . Это **примерно столько, сколько** мы ожидаем. Значит, система сплитования **работает корректно**

# Задание 2

Пришло время проанализировать результаты эксперимента, который мы провели вместе с командой дата сайентистов. Эксперимент проходил с 2025-01-03 по 2025-01-09 включительно. Для эксперимента были задействованы 2 и 1 группы. 

В группе 2 был использован один из новых алгоритмов рекомендации постов, группа 1 использовалась в качестве контроля. 

Основная гипотеза заключается в том, что новый алгоритм во 2-й группе приведет к увеличению CTR. 

Ваша задача — проанализировать данные А/B-теста. 

1. Выбрать метод анализа и сравнить CTR в двух группах (мы разбирали t-тест, Пуассоновский бутстреп, тест Манна-Уитни, t-тест на сглаженном ctr (α=5) а также t-тест и тест Манна-Уитни поверх бакетного преобразования).
2. Сравните данные этими тестами. А еще посмотрите на распределения глазами. Почему тесты сработали именно так?
3. Опишите потенциальную ситуацию, когда такое изменение могло произойти. Тут нет идеального ответа, подумайте.
4. Напишите рекомендацию, будем ли мы раскатывать новый алгоритм на всех новых пользователей или все-таки не стоит. При выполнении задания важно обосновать и аргументировать ваш вывод.

Выгрузим необходимые данные за нужный период
``` Python
q_1 = """
SELECT exp_group, 
    user_id,
    sum(action = 'like') as likes,
    sum(action = 'view') as views,
    likes/views as ctr
FROM {db}.feed_actions 
WHERE toDate(time) between '2025-01-03' and '2025-01-09'
    and exp_group in (2,1)
GROUP BY exp_group, user_id
"""
df_1 = ph.read_clickhouse(q_1, connection=connection)
```
Создадим переменные с исследуемыми группами. control - контрольная группа 1, treatment  - тестовая группа 2.

``` Python    
control = df_1[df_1.exp_group == 1].ctr
treatment = df_1[df_1.exp_group == 2].ctr
```
# 2.1 Проведем разведывательный анализ
``` Python 
# Сделаем графики в seaborn, с распределением наших CTR из контрольной и тестовой группы
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.tight_layout()

groups = sns.histplot(data = df_1, 
              x='ctr', 
              hue='exp_group', 
              palette = ['r', 'b'],
              alpha=0.5,
              kde=False)
```
<div style="text-align: center;">
<img src="https://sun9-80.userapi.com/impg/EhsknMqtQoLkQC_4DO4koCNl6GKb0NugrAmsbw/NIFxCzkpwFA.jpg?size=448x285&quality=95&sign=852f13dfbcb2c3cd98aa562700dbd2d6&type=album" alt="p" style="display: inline-block;">
</div>

Распределение группы control является вполне нормальным, имеющим "куполообразный" вид. В свою очередь распределение CTR в группе treatment, являяется бимодальным, т.к. имеет несколько максимумов. Исходя из этого, предпочтительными будут являтся тесты не основанные на нормльном распределении значений, поэтому для сравнения данных групп лучше подойдут бутстреп и Тест Манна-Уитни.

Посмотрим на средние значения, моды групп, а также проведем тест Шапиро-Уилка на нормальность распределения.
``` Python
print(f"Cреднее в группе control = {round(control.mean(),5)}\n" +
      f"Медиана в группе control = {round(control.median(),5)}")
```
Cреднее в группе control = 0.21677

Медиана в группе control = 0.20588
``` Python
print(f"Cреднее в группе treatment = {round(treatment.mean(),5)}\n" +
      f"Медиана в группе treatment = {round(treatment.median(),5)}")
```
Cреднее в группе treatment = 0.2161

Медиана в группе treatment = 0.15328

Из-за ненормальности распределения, мода и среднее значение в тестовой группе сильно разнятся. Медиана в тестовой группе меньше чем в контрольной, что может говорить об ухудшении CTR в данной группе
``` Python
# Тест Шапиро-Уилка для контрольной группы
stat, p = shapiro(control)
print(f"Shapiro-Wilk Test: Statistic = {stat}, P-value = {p}")
```
Shapiro-Wilk Test: Statistic = 0.9518704414367676, P-value = 0.0
``` Python
# Тест Шапиро-Уилка для тестовой группы
stat, p = shapiro(treatment)
print(f"Shapiro-Wilk Test: Statistic = {stat}, P-value = {p}")
```
Shapiro-Wilk Test: Statistic = 0.8931446075439453, P-value = 0.0

Хоть нам и казалось распределение в контрольной группе нормальным и даже с учетом, того, что статистика теста близка к 1, p-значение 0.0 указывает на то, что данные значительно отклоняются от нормального распределения в обоих группах.

# 2.2 Проверка гипотезы

## 2.2.1 t - тест

Сформулируем нулевую и альтернативную гипотезу для t-теста.

Н0 - CTR treatment группы меньше или равен CTR группы control.

Н1 - CTR treatment группы больше CTR группы control.

``` Python
t_statistic, p_value = stats.ttest_ind(treatment,control, equal_var=False, alternative='greater')  
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Statistic = {t_statistic}, P-value = {p_value}")

# Интерпретация
alpha = 0.05
if p_value < alpha:
    print("Отклоняем нулевую гипотезу: CTR  treatment группы больше CTR группы control.")
else:
    print("Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равен CTR группы control.")
```
Statistic = -0.4051491913112757, P-value = 0.6573133344296245

Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равен CTR группы control.

## 2.2.2 U-критерий Манна-Уитни

Сформулируем нулевую и альтернативную гипотезу для U-критерия Манна-Уитни.

Н0 - CTR treatment группы меньше или равно CTR группы control.

Н1 - CTR treatment группы больше CTR группы control.

``` Python
stat, p = mannwhitneyu(treatment, control, alternative='greater')
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Mann-Whitney U Test: Statistic = {stat}, P-value = {p}")

# Интерпретация
alpha = 0.05
if p < alpha:
    print("Отклоняем нулевую гипотезу: CTR  treatment группы больше CTR группы control.")
else:
    print("Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равно CTR группы control.")
```
Mann-Whitney U Test: Statistic = 43777627.0, P-value = 1.0

Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равно CTR группы control.

## 2.2.3 Пуассоновский бутстреп

``` Python
likes_control = df_1[df_1.exp_group == 1].likes.to_numpy()
views_control = df_1[df_1.exp_group == 1].views.to_numpy()
likes_treatment = df_1[df_1.exp_group == 2].likes.to_numpy()
views_treatment = df_1[df_1.exp_group == 2].views.to_numpy()

def bootstrap(likes_control, views_control, likes_treatment, views_treatment, n_bootstrap=2000):

    poisson_bootstraps1 = stats.poisson(1).rvs(
        (n_bootstrap, len(likes_control))).astype(np.int64)

    poisson_bootstraps2 = stats.poisson(1).rvs(
            (n_bootstrap, len(likes_treatment))).astype(np.int64)
    
    epsilon = 1e-9  # Маленькая константа, чтобы избежать деления на ноль
    globalCTR1 = (poisson_bootstraps1*likes_control).sum(axis=1) / ((poisson_bootstraps1*views_control).sum(axis=1) + epsilon)
    globalCTR2 = (poisson_bootstraps2*likes_treatment).sum(axis=1) / ((poisson_bootstraps2*views_treatment).sum(axis=1) + epsilon)

    return globalCTR1, globalCTR2
ctr_control, ctr_treatment = bootstrap(likes_control, views_control, likes_treatment, views_treatment)
sns.histplot(ctr_control)
sns.histplot(ctr_treatment)
```
<div style="text-align: center;">
<img src="https://sun9-14.userapi.com/impg/uvl8LwE2Yp6npzvJKb0w2rCcinmkf_EhiMIYcQ/Cv9dV8seroE.jpg?size=443x295&quality=95&sign=09a7245b22532e95daa94fc50d93eaf2&type=album" alt="p" style="display: inline-block;">
</div>

### Разница между глобальными CTR
sns.histplot(ctr_treatment - ctr_control)

<div style="text-align: center;">
<img src="https://sun9-23.userapi.com/impg/fhOzQDHU4-fQVIv8CmaOx93oJZkdiHrFI6rzbQ/RKU0wtP_GaU.jpg?size=420x268&quality=95&sign=341cfbed0157225cff289a036c7ae4af&type=album" alt="p" style="display: inline-block;">
</div>

Распределение разницы значений между контрольной группой и тестовой, говорит об отсутствии сходства групп и об отрицательном эффекте в тестовой группе, для того чтобы в этом убедится однозначно, проведем сравнение получившихся распределени t-тестом и U-критерием Манна-Уитни.

``` Python
t_statistic, p_value = stats.ttest_ind(ctr_treatment,ctr_control, equal_var=False, alternative='greater')  
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Statistic = {t_statistic}, P-value = {p_value}")

# Интерпретация
alpha = 0.05
if p_value < alpha:
    print("Отклоняем нулевую гипотезу: CTR  treatment группы больше CTR группы control")
else:
    print("Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равен CTR группы control")
```
Statistic = -265.52269714619854, P-value = 1.0
Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равен CTR группы control

``` Python
stat, p = mannwhitneyu(ctr_treatment, ctr_control, alternative='greater')
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Mann-Whitney U Test: Statistic = {stat}, P-value = {p}")

# Интерпретация
alpha = 0.05
if p < alpha:
       print("Отклоняем нулевую гипотезу: CTR  treatment группы больше CTR группы control")
else:
    print("Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равен CTR группы control")
```
Mann-Whitney U Test: Statistic = 0.0, P-value = 1.0

Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равен CTR группы control

Как и ожидалось, статистические тесты подтвердили предположение, о том что значения CTR тестовой группы равен или меньше, чем  CTR в контрольной.

# 2.2.4 Бакетное преобразование

Извлечем данные и разобъем пользователей на 50 бакетов (групп)

``` Python
q_bucket = """

SELECT exp_group, bucket,
    sum(likes)/sum(views) as bucket_ctr
FROM (SELECT exp_group, 
        xxHash64(user_id)%50 as bucket,
        user_id,
        sum(action = 'like') as likes,
        sum(action = 'view') as views,
        likes/views as ctr
    FROM {db}.feed_actions 
    WHERE toDate(time) between '2025-01-03' and '2025-01-09'
        and exp_group in (1,2)
    GROUP BY exp_group, bucket, user_id)
GROUP BY exp_group, bucket
"""
df_bucket = ph.read_clickhouse(q_bucket, connection=connection)
treatment_bucket_CTR = df_bucket[df_bucket.exp_group == 1].bucket_ctr
control_bucket_CTR = df_bucket[df_bucket.exp_group == 2].bucket_ctr
#тест Манна-Уитни видит отличие
stats.mannwhitneyu(treatment_bucket_CTR, control_bucket_CTR, alternative='greater')
print(f"Mann-Whitney U Test: Statistic = {stat}, P-value = {p}")

# Интерпретация
alpha = 0.05
if p < alpha:
    print("Отклоняем нулевую гипотезу: CTR  treatment группы больше CTR группы control.")
else:
    print("Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равно CTR группы control.")
```
Mann-Whitney U Test: Statistic = 0.0, P-value = 1.0

Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равно CTR группы control.

``` Python
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.tight_layout()

groups = sns.histplot(data = df_bucket, 
              x='bucket_ctr', 
              hue='exp_group', 
              palette = ['r', 'b'],
              alpha=0.5,
              kde=False)
```
<div style="text-align: center;">
<img src="https://sun9-2.userapi.com/impg/ritV7T2uC_eCFVt8vnmivvKh8FXHUsvfMb3KqA/3wlNqw-_ugQ.jpg?size=728x509&quality=95&sign=097afa06ca445499266ecc6622958f5e&type=album" alt="p" style="display: inline-block;">
</div>

# 2.2.5 Сглаженный CTR

Применим "сглаживание" CTR для более корректной оценки CTR пользователей с низким количеством просмотров.
``` Python
def get_smothed_ctr(user_likes, user_views, global_ctr, alpha):
    smothed_ctr = (user_likes + alpha * global_ctr) / (user_views + alpha)
    return smothed_ctr

global_ctr_control = df_1[df_1.exp_group == 1].likes.sum()/df_1[df_1.exp_group == 1].views.sum()
global_ctr_treatment = df_1[df_1.exp_group == 2].likes.sum()/df_1[df_1.exp_group == 2].views.sum()

group1 = df_1[df_1.exp_group == 1].copy()
sns.distplot(group1.ctr, 
             kde = False)

```
<div style="text-align: center;">
<img src="https://sun9-41.userapi.com/impg/thR8ZuaSVoH9rj5ByiMskf33wqq3zB_xTu8U9A/XoecbsIRPbo.jpg?size=705x506&quality=95&sign=1d781b394f5f841655efab1d0350a94f&type=album" alt="p" style="display: inline-block;">
</div>

``` Python
group1['smothed_ctr'] = df_1.apply(
    lambda x: get_smothed_ctr(x['likes'], x['views'], global_ctr_control, 5), axis=1)
sns.distplot(group1.smothed_ctr, 
             kde = False)
```
<div style="text-align: center;">
<img src="https://sun9-13.userapi.com/impg/qX8ToGN_XU8CNZD9sr0KxKpvF-dS74G8tOPv1Q/XGJZ1Tf0KGs.jpg?size=702x506&quality=95&sign=b14e49f2c7e1f3efe08624a336437bcf&type=album" alt="p" style="display: inline-block;">
</div>

``` Python
group2 = df_1[df_1.exp_group == 2].copy()
sns.distplot(group2.ctr, 
             kde = False)
```
<div style="text-align: center;">
<img src="https://sun9-59.userapi.com/impg/4mQq6HNDULYuTBDfRoBUcLJYmuxraU4PFw9xKw/F4tNygIuvo0.jpg?size=706x506&quality=95&sign=857b2a78a92bbbb408361cca56655744&type=album" alt="p" style="display: inline-block;">
</div>

``` Python
group2['smothed_ctr'] = df_1.apply(
    lambda x: get_smothed_ctr(x['likes'], x['views'], global_ctr_treatment, 5), axis=1)
sns.distplot(group2.smothed_ctr, 
             kde = False)
```
<div style="text-align: center;">
<img src="https://sun9-79.userapi.com/impg/21EcKRZE2bwyG8AWKTY24onPmJ1jK-Y1E9mn5Q/0Yc6Ueqsjf8.jpg?size=717x511&quality=95&sign=f2b7f3fcf1f8a880b2d52154f7244e40&type=album" alt="p" style="display: inline-block;">
</div>

``` Python
group1_ctr = group1.ctr
group2_ctr = group2.ctr

t_statistic, p_value = stats.ttest_ind(group2_ctr,group1_ctr, equal_var=False, alternative='greater')  
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Statistic = {t_statistic}, P-value = {p_value}")

# Интерпретация
alpha = 0.05
if p_value < alpha:
    print("Отклоняем нулевую гипотезу: CTR  treatment группы больше чем вгруппе control.")
else:
    print("Не удалось отклонить нулевую гипотезу:  CTR  treatment группы меньше или равно СTR группы control.")
```

Statistic = -0.4051491913112757, P-value = 0.6573133344296245

Не удалось отклонить нулевую гипотезу:  CTR  treatment группы меньше или равно СTR группы control.

``` Python
stat, p = mannwhitneyu(group2_ctr, group1_ctr, alternative='greater')
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Mann-Whitney U Test: Statistic = {stat}, P-value = {p}")

# Интерпретация
alpha = 0.05
if p < alpha:
    print("Отклоняем нулевую гипотезу:  CTR  treatment группы больше чем вгруппе control.")
else:
    print("Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равно СTR группы control.")
```
Mann-Whitney U Test: Statistic = 43777627.0, P-value = 1.0

Не удалось отклонить нулевую гипотезу: CTR  treatment группы меньше или равно СTR группы control.

# 2.3 Выводы

Нам неудалось отвергнуть нулевую гипотезу (Н0), все тесты показали что CTR treatment группы, меньше или равно СTR группы control. Следовательно новый алгоритм нуждается в доработке и не рекомендуется к внедрению в продукт, так как статистически доказано, что он не дает положительного влияния на CTR пользователей.

``` Python
df_1[df_1.exp_group == 1].likes.sum()
```
Результат: 140339
``` Python
df_1[df_1.exp_group == 2].likes.sum()
```
Результат: 132056

``` Python
t_statistic, p_value = stats.ttest_ind(df_1[df_1.exp_group == 2].likes, df_1[df_1.exp_group == 1].likes, equal_var=False)  
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Statistic = {t_statistic}, P-value = {p_value}")

# Интерпретация
alpha = 0.05
if p_value < alpha:
    print("Отклоняем нулевую гипотезу: разница в лайках между группами статистически значима")
else:
    print("Не удалось отклонить нулевую гипотезу:  разница в лайках между группами статистически не значима")
```
Statistic = -4.118069366090456, P-value = 3.836606381078642e-05

Отклоняем нулевую гипотезу: разница в лайках между группами статистически значима

``` Python
df_1[df_1.exp_group == 1].views.sum()
```
Результат: 669543

``` Python
df_1[df_1.exp_group == 2].views.sum()
```
Результат: 659454

``` Python
t_statistic, p_value = stats.ttest_ind(df_1[df_1.exp_group == 2].views, df_1[df_1.exp_group == 1].views, equal_var=False)  
#Параметр alternative='greater'- проверяем одностороннюю гипотезу о том, что значения в первой группе статистически значимо больше,
#чем во второй группе.
print(f"Statistic = {t_statistic}, P-value = {p_value}")

# Интерпретация
alpha = 0.05
if p_value < alpha:
    print("Отклоняем нулевую гипотезу: разница в лайках между группами статистически значима")
else:
    print("Не удалось отклонить нулевую гипотезу:  разница в лайках между группами статистически не значима") 

```

Statistic = -0.08399232327239431, P-value = 0.9330633806926314

Не удалось отклонить нулевую гипотезу:  разница в лайках между группами статистически не значима

Предположительно причина в нерелевантном контенте, который предлагает новый алгоритм, пользователям не интересны предложенные посты, темы и категории этих постов, от этого они меньше ставят лайков, при одинаковом количестве просмотров.

