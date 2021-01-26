# loading all required packages
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import os
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
import pickle
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


filename = 'data_text_preprocessed'
if os.path.isfile('data_text_preprocessed'):

    # load the model from disk

    loaded_model = pickle.load(open(filename, 'rb'))
else:
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
            model = data_text_preprocess(row["TEXT"], index, 'TEXT')

    pickle.dump(model, open(filename, 'wb'))

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

result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']

# let check again if there is any missing values
print(result[result.isnull().any(axis=1)])

# Creating the Training, Test and Validation Data
# Before splitting the data int training, test and validation data set.
y_true = result["Class"].values
result['Gene'] = result["Gene"].str.replace('\s+','_')
result['Variation'] = result['Variation'].str.replace('\s+','_')

# Splitting the data into train and test set
X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)
# split the train data now into train validation and cross validation
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
print('Number of data points in train data:', train_df.shape[0])
print('Number of data points in test data:', test_df.shape[0])
print('Number of data points in cross validation data:', cv_df.shape[0])

train_class_distribution = train_df['Class'].value_counts().sort_index()
test_class_distribution = test_df['Class'].value_counts().sort_index()
cv_class_distribution = cv_df['Class'].value_counts().sort_index()

print('Train class distribution', train_class_distribution)
print('Test class distribution', test_class_distribution)
print('CV class distribution', cv_class_distribution)
"""
So, what does above variable suggest us. This means in my train dataset we have 1 values with count of 363
class 2 values having count of 289 and so on.
It will be better idea to visualise it in graph format
"""

my_colors = {'r','g','b','k','y','m','c'}
train_class_distribution.plot(kind='bar',color=my_colors)
plt.xlabel('Class')
plt.ylabel(' Number of Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
# Let's look at distribution in form of percentage

sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')


my_colors = {'r','g','b','k','y','m','c'}
test_class_distribution.plot(kind='bar', color=my_colors)
plt.xlabel('Class')
plt.ylabel('Number of Data points per Class')
plt.title('Distribution of yi in test data')
plt.grid()
plt.show()

# Let's look at distribution in form of percentage
sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')


my_colors = {'r','g','b','k','y','m','c'}
cv_class_distribution.plot(kind='bar',color=my_colors)
plt.xlabel('Class')
plt.ylabel('Data points per Class')
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

# BUILDING A RANDOM MODEL
"""
Ok, so we need to generate 9 random numbers because we have 9 classes such that their sum must be equal to 
1 because sum of probability of all 9 classes must be equivalent to 1
"""
test_data_len = test_df.shape[0]
cv_data_len = cv_df.shape[0]

# we create a output array that has exactly same size as the CV data
cv_predicted_y = np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,9)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))

# Test-Set error.
#we create a output array that has exactly same as the test data
test_predicted_y = np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,9)
    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))

# Lets get the index of max probablity
predicted_y =np.argmax(test_predicted_y, axis=1)

# Lets see the output. these will be 665 values present in test dataset
print(predicted_y)

"""So you can see the index value ranging
from 0 to 8. So, lets make it as 1 to 9 we will increase this value by 1."""

predicted_y = predicted_y + 1

C = confusion_matrix(y_test, predicted_y)
labels = [1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(20,7))
sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

B =(C/C.sum(axis=0))
plt.figure(figsize=(20,7))
sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

A =(((C.T)/(C.sum(axis=1))).T)
plt.figure(figsize=(20,7))
sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

# Evaluating Gene Column
"""Now we look at each independent column to make sure its relevant for my target variable 
but the question is, how? Let's understand with our first column Gene which is categorical in nature.
Let's explore column Gene and lets look at its distribution 
"""

unique_genes = train_df['Gene'].value_counts()
print("Number of Unique Genes:", unique_genes.shape[0])
# The top ten genes that occurred most
print(unique_genes.head(10))

# Lets look at the cummulative distribution of unique Genes values
s = sum(unique_genes.values)
h = unique_genes.values/s
c = np.cumsum(h)
plt.plot(c, label = "Cumulative distribution of Genes")
plt.grid()
plt.legend()
plt.show()

"""So, now we need to convert these cetegorical variable to appropriate format which the machine learning
algorithm will be able to take as an input.
So we have 2 techniques to deal with it.
1. One hot encoding 
2. Response Encoding (Mean imputation)
let's use both of them to see which one works the best. So let's start encoding using one hot encoder
"""

# One hot encoding of Gene feature.
gene_vectorizer = CountVectorizer()
train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])

# Let's check the number of column generated after one hot encoding. One hot encoding will always return
# higher number of column.
print(train_gene_feature_onehotCoding.shape)
print(gene_vectorizer.get_feature_names())

"""
 Now lets create a response coding with Laplace smoothing.
 alpha : used for laplace smoothing
 feature : ['gene','variation']
 df: ['train_df','test_df','cv_df']
 algorithm
 _______________________________
 Consider all unique values and the number of occurrences of given feature in train data dataframe.
 build a vector (1*9), the first element = (number of times it occured in class1 + 10*alpha/
 number of time it occurred in total data+90*alpha)
 for a value of feature in df:
 if it is in train data:
 we add the vector that was scored in 'gv_dict' look up table to 'gv_fea'
 if it is not there is train:
 we add [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9] to 'gv_fea'
 return 'gv_fea' 
 ------------------------------------
"""

# get_gv_fea_dict: Get Gene variation Feature Dict
# get_gv_fea_dict: Get Gene varaition Feature Dict
def get_gv_fea_dict(alpha, feature, df):
    #value_count: it contains a dict like
    print(train_df['Gene'].value_counts())

    value_count = train_df[feature].value_counts()

    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation
    gv_dict = dict()

    # denominator will contain the number of time that particular feature occurred in whole data
    for i, denominator in value_count.items():
        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to particular class
        # vec is 9 diamensional vector
        vec = []
        for k in range(1, 10):
            print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])

            cls_cnt = train_df.loc[(train_df['Class'] == k) & (train_df[feature] == i)]

            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data
            vec.append((cls_cnt.shape[0] + alpha * 10) / (denominator + 90 * alpha))

        # we are adding the gene/variation to the dict as key and vec as value
        gv_dict[i] = vec
        print(gv_dict)
    return gv_dict


# Get Gene variation feature
def get_gv_feature(alpha, feature, df):

    gv_dict = get_gv_fea_dict(alpha, feature, df)
    # value_count is similar in get_gv_fea_dict
    value_count = train_df[feature].value_counts()

    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data
    gv_fea = []
    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea
    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea
    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9])
    #             gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])
    return gv_fea

# response-coding of the Gene feature
#alpha is used for Laplace smoothing
alpha = 1
# train gene feature
train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", train_df))
# Test gene feature
test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", test_df))
# Cross Validation gene feature
cv_gene_feature_responseCoding = np.array((get_gv_feature(alpha,"Gene", cv_df)))

# Let's look at columns after applying response encoding. We must be having 9 columns for Gene column after
# response encoding
print(train_gene_feature_responseCoding.shape)

"""
Now, question is how good is Gene column feature to predict my 9 classes. One idea could be that we will
build model having only gene column with one hot encoder with simple model like Logistic regression.
If log loss with only one column Gene comes out to be better than random model, than this feature is important.
"""

# We need a hyperparameter for the SGD classifier.
alpha = [10** x for x in range(-5,1)]

# We will be using SGD classifier
# we will also be using Caliberated Classifier to get the result into probability format to be used for
# for log loss .
cv_log_error_array = []
for i in alpha:
    clf = SGDClassifier(alpha=i,penalty = 'l2', loss='log', random_state=42)
    clf.fit(train_gene_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method = 'sigmoid')
    sig_clf.fit(train_gene_feature_onehotCoding,y_train)
    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
    cv_log_error_array.append(log_loss(y_cv,predict_y,labels = clf.classes_,eps=1e-15))
    print("For values of alpha =", i, 'The log loss is', log_loss(y_cv, predict_y, labels = clf.classes_, eps=1e-15))

# Let's plot the same to check the best Alpha value
fig,ax = plt.subplots()
ax.plot(alpha,cv_log_error_array, c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i], np.round(txt,3)), (alpha[i], cv_log_error_array[i]))
    plt.grid()
plt.title("Cross validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()

# Lets use best alpha value as we can see from above graph and compute log loss
best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_gene_feature_onehotCoding, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_gene_feature_onehotCoding, y_train)

predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

test_coverage=test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]
cv_coverage=cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

print('1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)
print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)

"""
Evaluating Variation column
Variation is also a categorical variable so we have to deal in same way like we have done for Gene column. We will again get the one hot encoder and response enoding variable for variation column.
"""
unique_variations = train_df['Variation'].value_counts()
print('Number of Unique Variations :', unique_variations.shape[0])
# the top 10 variations that occured most
print(unique_variations.head(10))

s = sum(unique_variations.values)
h = unique_variations.values/s
c = np.cumsum(h)
print(c)
plt.plot(c,label='Cumulative distribution of Variations')
plt.grid()
plt.legend()
plt.show()

# one-hot encoding of variation feature.
variation_vectorizer = CountVectorizer()
train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])

# Lets look at shape of one hot encoder column for variation
print(train_variation_feature_onehotCoding.shape)

# alpha is used for laplace smoothing
alpha = 1
# train gene feature
train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", train_df))
# test gene feature
test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df))
# cross validation gene feature
cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df))

# Lets look at the shape of this response encoding result
print(train_variation_feature_responseCoding.shape)

# Lets again build the model with only column name of variation column
# We need a hyperparemeter for SGD classifier.
alpha = [10 ** x for x in range(-5, 1)]

# We will be using SGD classifier
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# We will also be using Calibrated Classifier to get the result into probablity format t be used for log loss
cv_log_error_array = []
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_variation_feature_onehotCoding, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_variation_feature_onehotCoding, y_train)
    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:", log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

# Lets plot the same to check the best Alpha value
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()

best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_variation_feature_onehotCoding, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_variation_feature_onehotCoding, y_train)

predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

test_coverage=test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]
cv_coverage=cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

print('1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)
print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)



