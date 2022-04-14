# Тестирование алгоритмов бустинга



```python
# Импорт основных библиотек
import numpy as np
import pandas as pd

# Импорт библиотеки машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Импорт библиотек построения графиков и диаграмм
from matplotlib import pyplot as plt
import seaborn as sns

# Указание режима отображения диаграмм
%matplotlib inline

# Настройка параметров среды Pandas
pd.set_option("display.max_columns", 200)
```
### Загрузка исходных данных
```python
# Загрузка исходных данных об оттоке клиентов в компании Telcom
telcom_df = pd.read_csv('telco-customer-churn.csv')

# Вывод загруженных данных
telcom_df.head()
```
![png](Images/table_01.jpg)

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```


