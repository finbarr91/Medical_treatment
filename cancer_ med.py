# loading all required packages
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
from mlxtend.classifier import StackingClassifier

# Loading the dataset
data_variants = pd.read_csv('training_variants')
# Loading the training_text dataset. This is separated by //
data_text = pd.read_csv('training_text', sep='\|\|', engine = 'python', names = ['ID','TEXT'],skiprows=1)

print(data_variants.head(3))
"""
Let's understand the above data.
There are 4 fields above:
ID : row id used to link the mutation to the clinical evidence.
Gene : The gene where the genetic mutation is located
Variation : The aminoacid change for this mutations
Class : the class value 1-9, this genetic mutation has been classified on
"""

print(data_variants.info())

print(data_variants.describe())

# Checking the dimensions of the data
print(data_variants.shape)

# Checking the columns of the above dataset
print(data_variants.columns)

# Let's explore the training data text
print(data_text.head(3))
print(data_text.info())
print(data_text.describe())