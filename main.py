import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.set_option('display.max_columns', None)
plt.rcParams.update({'figure.max_open_warning': 0})


################### Utilities ####################

def distplot(feature, frame, color='g'):
    plt.figure(figsize=(8, 3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color=color)
    
    
def title(text):
    print('\n' + f' {text} '.center(60, '#'))


################ Dataset loading #################
df = pd.read_csv('bank/bank.csv', delimiter=';')

y = df['y']
X = df.drop(columns='y')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Boxplot of duration, for deposit "yes" and "no"
plt.figure()
sns.catplot(x="y", y="duration", kind='box', data=df)
plt.show()

############## EDA / Pre-processing ##############
title('Training set info and null values')
print(X_train.info())
print('\n')
print(X_train.isnull().any())

output_mapping = {'no': 0, 'yes': 1}

y_train = y_train.replace(output_mapping)

categorical_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']

X_train_categorical = X_train[categorical_cols].copy()

# categorical to discrete
for col in categorical_cols:
    X_train_categorical[col], _ = pd.factorize(X_train_categorical[col], sort=True)
    
for feat in categorical_cols:
    plt.figure(figsize=(8, 3))
    sns.catplot(x=feat, kind='count', data=X_train)
    plt.show()

numerical_cols = list(set(X_train.columns.tolist()) - set(categorical_cols))

for feat in numerical_cols:
    distplot(feat, X_train, color='cyan')

# scaling of numerical features
scaler = StandardScaler().fit(X_train[numerical_cols].astype('float64'))
X_train_std = pd.DataFrame(
    scaler.transform(X_train[numerical_cols].astype('float64')), 
    columns=numerical_cols, 
    index=X_train[numerical_cols].index
)

X_train = pd.concat([X_train_categorical, X_train_std], axis=1, sort=False)

# Temporary dataframe for corr. matrix plot
temp_df = pd.concat([X_train, y_train], axis=1, sort=False)

plt.figure(figsize=(12, 10), dpi=80)
sns.heatmap(
    temp_df.corr(), 
    xticklabels=temp_df.corr().columns, 
    yticklabels=temp_df.corr().columns, 
    center=0, 
    annot=True,
    fmt='.2f',
    square=True,
    linewidths=.5
)

plt.title('Correlation Matrix', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Pairplot of correlated features
cols_to_plot = ['housing', 'poutcome', 'pdays', 'previous', 'y']
plt.figure(figsize=(12, 10), dpi=80)
sns.pairplot(temp_df[cols_to_plot], hue='y', corner=True)
plt.show()


##################### Models #####################

models = [
    LogisticRegression(solver='saga', class_weight='balanced'),
    KNeighborsClassifier(weights='distance'),
    DecisionTreeClassifier(class_weight='balanced'),
    CategoricalNB(alpha=1.0)  # uses Laplace Smoothing
]

models_names = [
    'Logistic Regression',
    'KNN',
    'DT',
    'Naive Bayes'
]

models_hparameters = [
    # C is the inverse of lambda, lower C -> stronger regularization
    {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]},
    
    {'n_neighbors': list(range(1, 16, 2))},  # KNN
    
    {'criterion': ['gini', 'entropy']},  # DT
    
    {'fit_prior': [True, False]} # NB
]

title('Grid Search')

chosen_hparameters = []
estimators = []

for model, model_name, hparameters in zip(models, models_names, models_hparameters):
    print('\n' + model_name)
    clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=5) # 'f1_weighted'
          
    if model_name == 'Naive Bayes':
        clf.fit(X_train_categorical, y_train)
    else:
        clf.fit(X_train, y_train)
    
    chosen_hparameters.append(clf.best_params_)
    estimators.append((model_name, clf))
    print('Accuracy:  ', clf.best_score_)
    
    for hparam in hparameters:
        print(f'\t--> best value for hyperparameter "{hparam}": ', clf.best_params_.get(hparam))
    
estimators.pop()    # removes NB from estimator list

title('Cross Validation for model performance estimation')

clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
scores = cross_validate(clf_stack, X_train, y_train, cv=5, scoring=('f1_weighted', 'accuracy'))
print('\n') 
print('The cross-validated weighted F1-score of the Stacking Ensemble is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Accuracy of the Stacking Ensemble is ', np.mean(scores['test_accuracy']))

knn_model = KNeighborsClassifier(n_neighbors=9)
scores = cross_validate(knn_model, X_train, y_train, cv=5, scoring=('f1_weighted', 'accuracy'))
print('\n')
print('The cross-validated weighted F1-score of KNN is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Accuracy of KNN is ', np.mean(scores['test_accuracy']))

dt_model = DecisionTreeClassifier(class_weight='balanced', criterion='gini')
scores = cross_validate(dt_model, X_train, y_train, cv=5, scoring=('f1_weighted', 'accuracy'))
print('\n')
print('The cross-validated weighted F1-score of DT is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Accuracy of DT is ', np.mean(scores['test_accuracy']))

nb_model = CategoricalNB(alpha=1.0, fit_prior=True)
scores = cross_validate(nb_model, X_train_categorical, y_train, cv=5, scoring=('f1_weighted', 'accuracy'))
print('\n')
print('The cross-validated weighted F1-score of Naive Bayes is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Accuracy of Naive Bayes is ', np.mean(scores['test_accuracy']))


################## Final model ###################
final_model = clf_stack


#################### Wrapper #####################
## Time consuming if using clf_stack as final model

# title('Feature selection')

# t0 = time.time()

# sfs = SequentialFeatureSelector(final_model, cv=2) 

# sfs.fit(X_train, y_train)
# print('\n ............. ')
# print('Selected features: ', sfs.get_support())
# print('# of selected features: ', sfs.get_support()[sfs.get_support()].size)
# print('\n ............. ')

# X_train = sfs.transform(X_train)

# print(f'\nFeature selection took {time.time()-t0} sec')

############## Training final model ##############

title('Training final model')

t0 = time.time()

final_model.fit(X_train, y_train)
# final_model.fit(X_train_categorical, y_train) # uncomment if using NB as final model

print(f'\nFinal model training took {time.time()-t0} sec')


#################### Testing #####################
################# Pre-processing #################
title('Testing set info and null values')
print(X_test.info())
print('\n')
print(X_test.isnull().any())

y_test = y_test.replace(output_mapping)

X_test_categorical = X_test[categorical_cols].copy()

# categorical to discrete:
for col in categorical_cols:
    X_test_categorical[col], _ = pd.factorize(X_test_categorical[col], sort=True)

X_test_std = pd.DataFrame(
    scaler.transform(X_test[numerical_cols].astype('float64')), 
    columns=numerical_cols, 
    index=X_test[numerical_cols].index
)

X_test = pd.concat([X_test_categorical, X_test_std], axis=1, sort=False)


######### Prediction and testing results #########

# X_test = sfs.transform(X_test)  # uncomment if using wrapper
y_pred = final_model.predict(X_test)
# y_pred = final_model.predict(X_test_categorical) # uncomment if using NB as final model

title('Final Testing RESULTS')
print('Accuracy is ', accuracy_score(y_test, y_pred))
print('Precision is ', precision_score(y_test, y_pred, average='weighted'))
print('Recall is ', recall_score(y_test, y_pred, average='weighted'))
print('F1-Score is ', f1_score(y_test, y_pred, average='weighted'))


############### Performance plots ################
# Comment the following code if using Naive Bayes as final_model

plt.figure()
plot_roc_curve(final_model, X_test, y_test)
plt.show()

plt.figure()
plot_confusion_matrix(final_model, X_test, y_test)
plt.show()