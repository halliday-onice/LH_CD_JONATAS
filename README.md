# LH_CD_JONATAS
**Repositório destinado ao desafio Lighthouse 1 sem 2024**

***Todo o desafio foi feito no jupyternotebook e é necessário ter instalado as seguintes bibliotecas:***

**Numpy**

**Pandas**

**Matplotlib**

**Seaborn**

**Plotly**

**SciPy**

**Mlxtend**

**NLTK**

E para isso é necessário instalá-las com o **pip** da seguinte forma:
```
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install plotly
pip install scipy
pip install mlxtend
pip install nltk
```
Para a importação é necessário executar as seguintes comandos:
``` 
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import plotly.express as px
%matplotlib inline
#for box-cox transformation
from scipy import stats
#for min-max scaling
from mlxtend.preprocessing import minmax_scaling
#NLP
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import nltk
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('punkt')
#Loading the libraries for prediction
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


warnings.simplefilter("ignore")
```
