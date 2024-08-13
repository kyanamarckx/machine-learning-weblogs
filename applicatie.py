# imports
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore')


# create function to read in data
def read_data(input_file):
    weblogs = pd.read_csv(input_file)
    print("Reading data ...")
    return weblogs


# create function to clean data
def clean_data(weblogs):
    # drop columns with a correlation value lower than 0.3
    keep = ['UNASSIGNED', 'STANDARD_DEVIATION', 'MAX_BARRAGE', 'CONSECUTIVE', 'DEPTH', 'STD_DEPTH', 'SF_REFERRER', 'HTML_TO_IMAGE', 'ROBOT']
    print("Cleaning data ...")
    weblogs_clean = weblogs.loc[:, keep]

    # find the outliers for UNASSIGNED and replace them with the median
    Q1 = weblogs_clean['UNASSIGNED'].quantile(0.25)
    Q3 = weblogs_clean['UNASSIGNED'].quantile(0.75)
    IQR = Q3 - Q1
    print("Removing outliers for UNASSIGNED ...")
    outliers = weblogs_clean[(weblogs_clean['UNASSIGNED'] < Q1 - 1.5 * IQR) | (weblogs_clean['UNASSIGNED'] > Q3 + 1.5 * IQR)]
    weblogs_clean.loc[outliers.index, 'UNASSIGNED'] = weblogs_clean['UNASSIGNED'].median()

    # find the outliers for STANDARD_DEVIATION and replace them with the median
    Q1 = weblogs_clean['STANDARD_DEVIATION'].quantile(0.25)
    Q3 = weblogs_clean['STANDARD_DEVIATION'].quantile(0.75)
    IQR = Q3 - Q1
    print("Removing outliers for STANDARD_DEVIATION ...")
    outliers = weblogs_clean[(weblogs_clean['STANDARD_DEVIATION'] < Q1 - 1.5 * IQR) | (weblogs_clean['STANDARD_DEVIATION'] > Q3 + 1.5 * IQR)]
    weblogs_clean.loc[outliers.index, 'STANDARD_DEVIATION'] = weblogs_clean['STANDARD_DEVIATION'].median()

    # find the outliers for MAX_BARRAGE and replace them with the median
    Q1 = weblogs_clean['MAX_BARRAGE'].quantile(0.25)
    Q3 = weblogs_clean['MAX_BARRAGE'].quantile(0.75)
    IQR = Q3 - Q1
    print("Removing outliers for MAX_BARRAGE ...")
    outliers = weblogs_clean[(weblogs_clean['MAX_BARRAGE'] < Q1 - 1.5 * IQR) | (weblogs_clean['MAX_BARRAGE'] > Q3 + 1.5 * IQR)]
    weblogs_clean.loc[outliers.index, 'MAX_BARRAGE'] = weblogs_clean['MAX_BARRAGE'].median()

    # find the outliers for STD_DEPTH and replace them with the median
    Q1 = weblogs_clean['STD_DEPTH'].quantile(0.25)
    Q3 = weblogs_clean['STD_DEPTH'].quantile(0.75)
    IQR = Q3 - Q1
    print("Removing outliers for STD_DEPTH ...")
    outliers = weblogs_clean[(weblogs_clean['STD_DEPTH'] < Q1 - 1.5 * IQR) | (weblogs_clean['STD_DEPTH'] > Q3 + 1.5 * IQR)]
    weblogs_clean.loc[outliers.index, 'STD_DEPTH'] = weblogs_clean['STD_DEPTH'].median()

    # find the outliers for SF_REFERRER and replace them with the median
    Q1 = weblogs_clean['SF_REFERRER'].quantile(0.25)
    Q3 = weblogs_clean['SF_REFERRER'].quantile(0.75)
    IQR = Q3 - Q1
    print("Removing outliers for SF_REFERRER ...")
    outliers = weblogs_clean[(weblogs_clean['SF_REFERRER'] < Q1 - 1.5 * IQR) | (weblogs_clean['SF_REFERRER'] > Q3 + 1.5 * IQR)]
    weblogs_clean.loc[outliers.index, 'SF_REFERRER'] = weblogs_clean['SF_REFERRER'].median()

    # find the outliers for HTML_TO_IMAGE and replace them with the median
    Q1 = weblogs_clean['HTML_TO_IMAGE'].quantile(0.25)
    Q3 = weblogs_clean['HTML_TO_IMAGE'].quantile(0.75)
    IQR = Q3 - Q1
    print("Removing outliers for HTML_TO_IMAGE ...")
    outliers = weblogs_clean[(weblogs_clean['HTML_TO_IMAGE'] < Q1 - 1.5 * IQR) | (weblogs_clean['HTML_TO_IMAGE'] > Q3 + 1.5 * IQR)]
    weblogs_clean.loc[outliers.index, 'HTML_TO_IMAGE'] = weblogs_clean['HTML_TO_IMAGE'].median()

    # fill in the missing values with the mean
    print("Filling in missing values ...")
    weblogs_clean = weblogs_clean.fillna(weblogs_clean.mean())

    # return the cleaned data
    return weblogs_clean


# create function to train the chosen model
def train_model(weblogs_clean):
    # set the X and y
    X = weblogs_clean.drop('ROBOT', axis=1)
    y = weblogs_clean['ROBOT']

    # split the data
    print("Splitting the data ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create the Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    # set the parameters
    param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15],
    }

    # create the grid search
    print("Tuning hyperparameters ...")
    gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', return_train_score=True)

    # fit the grid search
    gs.fit(X_train, y_train)

    # create the new Random Forest Classifier model
    print("Creating tuned model ...")
    rfc_tuned = RandomForestClassifier(n_estimators=gs.best_params_['n_estimators'], max_depth=gs.best_params_['max_depth'], random_state=42, n_jobs=-1)

    # create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', rfc_tuned)
    ])

    # fit the pipeline
    print("Fitting the data ...")
    pipeline.fit(X_train, y_train)

    # predict the values
    print("Predicting the values ...")
    pred_rfc_tuned = pipeline.predict(X_test)

    # predict the probabilities (percentage)
    print("Predicting the probabilities ...")
    pred_rfc_tuned_proba = pipeline.predict_proba(X_test)
    pred_rfc_tuned_proba_percentage = pred_rfc_tuned_proba * 100

    # add the predicted values to a file
    pred_rfc_tuned.tofile('project\predictions.csv', sep=',\n', format='%.0f')

    # reshape the probabilities array
    pred_rfc_tuned_proba_percentage = pred_rfc_tuned_proba_percentage.reshape(-1, 2)

    # add the predicted probabilities to a file, the first column is 'mens', the second is 'robot'
    np.savetxt('project\probabilities.csv', pred_rfc_tuned_proba_percentage, delimiter=',', fmt='%.2f')

    # return y_test and pred_rfc_tuned for the confusion matrix
    return X_train, y_train, y_test, pipeline, pred_rfc_tuned


# create function to evaluate the model
def evaluate_model(X_train, y_train, y_test, rfc_tuned, pred_rfc_tuned):
    # accuracy score
    print("Calculating the accuracy score ...")
    accuracy = accuracy_score(y_test, pred_rfc_tuned) * 100

    # cross validation score
    print("Calculating the cross validation score ...")
    cross_val = cross_val_score(rfc_tuned, X_train, y_train, cv=5).mean() * 100

    # precision score
    print("Calculating the precision score ...")
    precision = precision_score(y_test, pred_rfc_tuned) * 100

    # recall score
    print("Calculating the recall score ...")
    recall = recall_score(y_test, pred_rfc_tuned) * 100

    # f1 score
    print("Calculating the f1 score ...")
    f1 = f1_score(y_test, pred_rfc_tuned) * 100

    # r2 score
    print("Calculating the r2 score ...")
    r2 = r2_score(y_test, pred_rfc_tuned) * 100

    # return the scores
    return accuracy, cross_val, precision, recall, f1, r2


# create function to get the values from the confusion matrix
def get_values_confusionMatrix(y_test, pred_rfc_tuned):
    # create the confusion matrix
    print("Creating the confusion matrix ...")
    cm = confusion_matrix(y_test, pred_rfc_tuned)

    # get the true positives
    tn = cm[0][0]

    # get the false positives
    fp = cm[0][1]

    # get the false negatives
    fn = cm[1][0]

    # get the true negatives
    tp = cm[1][1]

    # return the values
    return cm, tn, fp, fn, tp


# create function to plot and save the confusion matrix
def plot_confusionMatrix(cm, rfc_tuned):
    print("Plotting the confusion matrix ...")
    cm_rfc = ConfusionMatrixDisplay(cm, display_labels=rfc_tuned.classes_)
    cm_rfc.plot(cmap='Blues', values_format='d')
    plt.savefig('project\confusion_matrix.png')


# create a function to call all the functions
def main(input):
    weblogs = read_data(input)
    weblogs_clean = clean_data(weblogs)
    X_train, y_train, y_test, rfc_tuned, pred_rfc_tuned = train_model(weblogs_clean)
    accuracy, cross_val, precision, recall, f1, r2 = evaluate_model(X_train, y_train, y_test, rfc_tuned, pred_rfc_tuned)
    cm, tn, fp, fn, tp = get_values_confusionMatrix(y_test, pred_rfc_tuned)
    plot_confusionMatrix(cm, rfc_tuned)

    # print the scores
    print('Accuracy Score:', accuracy, '%')
    print('Cross Validation Score:', cross_val, '%')
    print('Precision Score:', precision, '%')
    print('Recall Score:', recall, '%')
    print('F1 Score:', f1, '%')
    print('R2 Score:', r2, '%')

    # print the values from the confusion matrix
    print('True Negatives:', tn)
    print('False Positives:', fp)
    print('False Negatives:', fn)
    print('True Positives:', tp)


# call the main function
inputfile = 'project/weblogs.csv'
inputfile = str(input("Enter the name of the csv file: "))
main(inputfile)
