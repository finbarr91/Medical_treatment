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
print(data_variants.Class.unique())

# We would like to remove all stopwords like a, is, an, the, ......
# so we collecting all of them from nltk library

stop_words = set(stopwords.words('english'))

def data_text_preprocess (total_text, ind, col):
    # Remove int values from the text data as that might not be important
    if type(total_text) is not int:
        string = " "
        # replacing all the special character with spaces
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+', ' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()

        for word in total_text.split():
            # if the word is not a stop word then retain that word from text
            if not word in stop_words:
                string += word+ " "
                data_text[col][ind] = string

# Below code will take some time because its huge text
for index, row in data_text.iterrows():
    if type(row['TEXT']) is str:
        data_text_preprocess(row["TEXT"], index, 'TEXT')

# Merging both the gene_variations and text data based on ID
result  = pd.merge(data_variants,data_text, on='ID', how='left')
print('\n Result:\n',result.head())

# It is very important to look for missing values else they create problem in the final analysis.
print(result[result.isnull().any(axis=1)])

'''
There are some missing data in the dataset. Now what do we do with this missing value. 
One way is to drop these rows having missing values or we can do some imputations on it.  
Let’s go with imputation only. But the question is what do we impute:
We can merge the gene and variation column, Let's do it.
'''

result.loc[result['TEXT'].isnull(),'TEXT'] = result['GENE']+ ' '+result['Variation']

# let check again if there is any missing values
print(result[result.isnull().any(axis=1)])

# Creating the Training, Test and Validation Data
# Before splitting the data int training, test and validation data set.
y_true = result["Class"].values
result['Gene'] = result["Gene"].str.replace('\s+','_')
result['Variation'] = result['Variation'].str.replace('\s+','_')

# Splitting the data into train and test set.
X_train, test_df,y_train,y_test = train_test_split(result,y_true, stratify=y_true,test_size = 0.2)
# Split the train data now into train validation and cross validation
train_df, cv_df, y_train, y_cv = train_test_split(X_train,y_train, stratify=y_train, test_size=0.2)
print('Number of data points in train data:', train_df.shape[0])
print('Number of data points in test data:', test_df.shape[0])
print('Number of data points in cross validation data:', cv_df.shape[0])

# Let's look at the distribution of data in train, test and validation set
train_class_distribution = train_df['Class'].value_counts().sortlevel()
test_class_distribution = test_df['Class'].value_counts().sortlevel()
cv_class_distribution = cv_df['Class'].value_counts().sortlevel()

print('Train class distribution', train_class_distribution)
print('Test class distribution', test_class_distribution)
print('CV class distribution', cv_class_distribution)
"""
So, what does above variable suggest us. This means in my train dataset we have 1 values with count of 363
class 2 values having count of 289 and so on.
It will be better idea to visualise it in graph format
"""
my_color = ['r','g','b','k','y','m','c']
sns.barplot(train_class_distribution,color=my_color) # Train distribution plot
plt.xlabel("Class")
plt.ylabel("Number of Data points per Class")
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

# Let's look at distribution in form of percentage

sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')



my_color = ['r','g','b','k','y','m','c']
sns.barplot(test_class_distribution,color=my_color) # Test distribution plot
plt.xlabel("Class")
plt.ylabel("Number of Data points per Class")
plt.title('Distribution of yi in test data')
plt.grid()
plt.show()

# Let's look at distribution in form of percentage
sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')

my_color = ['r','g','b','k','y','m','c']
sns.barplot(cv_class_distribution,color=my_color) # Cross validation distribution plot
plt.xlabel("Class")
plt.ylabel("Number of Data points per Class")
plt.title('Distribution of yi in cross validation data')
plt.grid()
plt.show()

# Let's look at distribution in form of percentage
sorted_yi = np.argsort(-cv_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',cv_class_distribution.values[i], '(', np.round((cv_class_distribution.values[i]/cv_df.shape[0]*100), 3), '%)')

"""
Now question is because we need log-loss as final evaluation metrics how do we say that model we are going
to build will be good model. For doing this we will build a random model and will evaluate log loss.
Our model should return lower log loss value than this. 
"""

