import math
import statistics
import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from statistics import *
from sklearn import svm
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, make_scorer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

random_seed = 1833



#################### Preprocessing ##################

class Data:
    def __init__(self, ID, swipe):
        self.ID = ID
        self.swipe = swipe
        self.X = []
        self.Y = []
        self.Pressure = []
        self.Area = []
        self.Time = []
        self.Velocity = []
        self.Acc = []
        self.features = []


def preprocess(data_path):
    file = open(data_path, 'r')
    data = []  # list of Examples
    header = file.readline().strip()  # Removing the header
    line = file.readline()

    # Initializing data object with one element
    tokens = line.strip().split(",")
    obj = Data(int(tokens[0]), int(tokens[1]))
    obj.X.append(float(tokens[2]))
    obj.Y.append(float(tokens[3]))
    obj.Pressure.append(float(tokens[4]))
    obj.Area.append(float(tokens[5]))
    obj.Time.append(float(tokens[6]))
    data.append(obj)
    i = 0

    # read the examples
    for line in file:
        tokens = line.strip().split(",")
        # reading data for each user separately
        if data[i].ID == int(tokens[0]):
            # combining individual swipe data for each user
            if data[i].swipe == int(tokens[1]):
                obj.X.append(float(tokens[2]))
                obj.Y.append(float(tokens[3]))
                obj.Pressure.append(float(tokens[4]))
                obj.Area.append(float(tokens[5]))
                obj.Time.append(float(tokens[6]))
            else:
                i += 1
                obj = Data(int(tokens[0]), int(tokens[1]))
                obj.X.append(float(tokens[2]))
                obj.Y.append(float(tokens[3]))
                obj.Pressure.append(float(tokens[4]))
                obj.Area.append(float(tokens[5]))
                obj.Time.append(float(tokens[6]))
                data.append(obj)
        else:
            i += 1
            obj = Data(int(tokens[0]), int(tokens[1]))
            obj.X.append(float(tokens[2]))
            obj.Y.append(float(tokens[3]))
            obj.Pressure.append(float(tokens[4]))
            obj.Area.append(float(tokens[5]))
            obj.Time.append(float(tokens[6]))
            data.append(obj)

    # record Velocity and Acc
    for obj in data:
        obj.Velocity = [math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / (t2 - t1) for x1, x2, y1, y2, t1, t2 in
                        zip(obj.X[:-1], obj.X[1:], obj.Y[:-1], obj.Y[1:], obj.Time[:-1], obj.Time[1:])] \
            if len(obj.X) > 1 else [0]
        obj.Acc = [(v2 - v1) / (t2 - t1) for v1, v2, t1, t2 in
                   zip(obj.Velocity[:-1], obj.Velocity[1:], obj.Time[:-1], obj.Time[1:])] \
            if len(obj.Velocity) > 1 else [0]
    return data


###################  Feature extraction #############
# Extract all the features from all the users and save it in a folder using the following folder structure
# ExtractedFeatures\<User>\<Phone-hold-style>\<Session1/Session2> or whatever is suitable
def extract_features(data_obj):
    """
            helper function to extract features
     """

    features = []
    # x-coordinate of the start-point (1)
    features.append(data_obj.X[0])
    # y-coordinate of the start-point (2)
    features.append(data_obj.Y[0])
    # x-coordinate of the endpoint (3)
    features.append(data_obj.X[-1])
    # y-coordinate of the endpoint (4)
    features.append(data_obj.Y[-1])

    # subtracting X values gives delta X
    X_deltas = [data_obj.X[i] - data_obj.X[i - 1] for i in range(1, len(data_obj.X))] if len(data_obj.X) > 1 else [0]
    # Average Change of X position (5)
    features.append(mean(X_deltas))
    # subtracting X values gives delta X
    Y_deltas = [data_obj.Y[i] - data_obj.Y[i - 1] for i in range(1, len(data_obj.Y))] if len(data_obj.Y) > 1 else [0]
    # Average Change of Y position (6)
    features.append(mean(Y_deltas))
    # Swipe Width (7)
    features.append(max(data_obj.X) - min(data_obj.X))
    # Swipe Height (8)
    features.append(max(data_obj.Y) - min(data_obj.Y))

    # creates a list of 4-tuples of form (x1,x2,y1,y2) and chooses the max slope
    max_slope = max([(y2 - y1) / (x2 - x1) if not x1 == x2 else 0 for x1, x2, y1, y2 in
                     zip(data_obj.X[:-1], data_obj.X[1:], data_obj.Y[:-1], data_obj.Y[1:])]) \
        if len(data_obj.X) > 1 else 0
    # Max Slope (9)
    features.append(max_slope)
    # creates a list of 4-tuples of form (x1,x2,y1,y2) and chooses the min slope
    min_slope = min([(y2 - y1) / (x2 - x1) if not x1 == x2 else 1000 for x1, x2, y1, y2 in
                     zip(data_obj.X[:-1], data_obj.X[1:], data_obj.Y[:-1], data_obj.Y[1:])]) \
        if len(data_obj.X) > 1 else 0
    # Min Slope (10)
    features.append(min_slope)

    # Toltal Duration (11)
    features.append(data_obj.Time[-1] - data_obj.Time[0])
    # mean; std; and 1st, 2nd, and 3rd percentiles for the velocity vector (12), (13), (14), (15), (16)
    features.append(mean(data_obj.Velocity))
    features.append(statistics.stdev(data_obj.Velocity) if len(data_obj.Velocity) > 1 else 0)
    # converting the list into np array to facilitate computing quartiles
    velocity_to_numpy = np.asarray(data_obj.Velocity)
    features.append(np.percentile(velocity_to_numpy, 25))
    features.append(np.percentile(velocity_to_numpy, 50))
    features.append(np.percentile(velocity_to_numpy, 75))

    # mean; std; and 1st, 2nd, and 3rd percentiles for the acceleration vector (17), (18), (19), (20), (21)
    features.append(mean(data_obj.Acc))
    features.append(statistics.stdev(data_obj.Acc) if len(data_obj.Acc) > 1 else 0)
    # converting the list into np array to facilitate computing quartiles
    acc_to_numpy = np.asarray(data_obj.Acc)
    features.append(np.percentile(acc_to_numpy, 25))
    features.append(np.percentile(acc_to_numpy, 50))
    features.append(np.percentile(acc_to_numpy, 75))

    # mean; std; and 1st, 2nd, and 3rd percentiles for the pressure vector (22), (23), (24), (25), (26)
    features.append(mean(data_obj.Pressure))
    features.append(statistics.stdev(data_obj.Pressure) if len(data_obj.Pressure) > 1 else 0)
    # converting the list into np array to facilitate computing quartiles
    pressure_to_numpy = np.asarray(data_obj.Pressure)
    features.append(np.percentile(pressure_to_numpy, 25))
    features.append(np.percentile(pressure_to_numpy, 50))
    features.append(np.percentile(pressure_to_numpy, 75))

    # mean; std; and 1st, 2nd, and 3rd percentiles for the area vector (27), (28), (29), (30), (31)
    features.append(mean(data_obj.Area))
    features.append(statistics.stdev(data_obj.Area) if len(data_obj.Area) > 1 else 0)
    # converting the list into np array to facilitate computing quartiles
    area_to_numpy = np.asarray(data_obj.Area)
    features.append(np.percentile(area_to_numpy, 25))
    features.append(np.percentile(area_to_numpy, 50))
    features.append(np.percentile(area_to_numpy, 75))

    # Number of Data Points (32)
    features.append(len(data_obj.X))

    # Computing Attack Angle: the angle through which the stroke starts
    # the angle is Pi/2 (1.5708) if x-coordinate doesn't change and 0 if we have 1 data point
    attack_angle = math.atan((data_obj.Y[1] - data_obj.Y[0]) / (data_obj.X[1] - data_obj.X[0])) \
        if not len(data_obj.X) < 2 and not data_obj.X[0] == data_obj.X[1] else 0 if len(data_obj.X) < 2 else 1.5708
    # Attack Angle (33)
    features.append(attack_angle)
    # Computing leaving Angle: the angle through which the stroke ends
    leaving_angle = math.atan((data_obj.Y[-1] - data_obj.Y[-2]) / (data_obj.X[-1] - data_obj.X[-2])) \
        if not len(data_obj.X) < 2 and not data_obj.X[-2] == data_obj.X[-1] else 0 if len(data_obj.X) < 2 else 1.5708

    # Leaving Angle (34)
    features.append(leaving_angle)

    # Mid-Stroke Area (35)
    features.append(data_obj.Area[len(data_obj.Area) // 2])
    # Mid-Stroke Pressure (36)
    features.append(data_obj.Area[len(data_obj.Pressure) // 2])

    # Assigning the list of features to the swipe data_obj
    data_obj.features = features


def select_features(training_X, training_y, genuine_testing, impostor_testing, thresholdK):
    fselector = SelectKBest(mutual_info_classif, k=int(thresholdK))
    ## Selecting k features based on mutuall information
    fselector.fit(training_X, training_y)
    training_X = fselector.transform(training_X)
    genuine_testing = fselector.transform(genuine_testing)
    impostor_testing = fselector.transform(impostor_testing)
    return training_X, training_y, genuine_testing, impostor_testing  # returning the matrix with selected features


def outliers(values):
    """
    Function to find the fraction of outliers in the data
    :param values: features values
    :return:  outlier fraction
    """
    outliers = 0
    Q1 = np.quantile(values, 0.25)
    Q3 = np.quantile(values, 0.75)
    IQR = Q3 - Q1
    for v in values:
        if v < (Q1 - 1.5 * IQR) or v > (Q3 + 1.5 * IQR):
            outliers += 1
    if outliers == 0:
        return 0.00001
    return outliers / len(values)


def get_error_rates(training_X, training_y, genuine_testing, impostor_testing, classification_method):
    genuine_y = get_labels(genuine_testing,
                           1)  # 1-D array of all ONES, number of elements shall be the number of rows in genuine_testing
    impostor_y = get_labels(impostor_testing,
                            0)  # 1-D array of all ZEROES, number of elements shall be the number of rows in impostor_testing
    testing_y = np.append(genuine_y, impostor_y)  # concatenate genuine_y and impostor_y

    if classification_method == "kNN":  # This is an example of how can you use kNN
        n_neighbors = [int(x) for x in range(5, 10, 1)]
        # print('n_neighbors',n_neighbors)
        dist_met = ['manhattan', 'euclidean']
        # create the random grid
        param_grid = {'n_neighbors': n_neighbors,
                      'metric': dist_met}
        CUAuthModel = KNeighborsClassifier()
        scoring_function = 'f1'  # You can use scoring function as HTER, see Aux_codes.py for details
        # Grid search for best parameter search .. using 10 fold cross validation and 'f1' as a scoring function.
        SearchTheBestParam = GridSearchCV(estimator=CUAuthModel, param_grid=param_grid, cv=10,
                                          scoring=scoring_function)
        SearchTheBestParam.fit(training_X, training_y)
        best_nn = SearchTheBestParam.best_params_['n_neighbors']
        best_dist = SearchTheBestParam.best_params_['metric']

        # Retraining the model again using the best parameter and testing, remember k = 1 will always give 100% accuracy :) on training data
        FinalModel = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist)
        FinalModel.fit(training_X, training_y)

        pred_gen_lables = FinalModel.predict(genuine_testing)
        pred_imp_lables = FinalModel.predict(impostor_testing)
        pred_testing_labels = np.concatenate((pred_gen_lables, pred_imp_lables))

        # computing the error rates for the current pred

        tn, fp, fn, tp = confusion_matrix(testing_y, pred_testing_labels).ravel()
        # far = fp / (fp + tn)
        # frr = fn / (fn + tp)
        # hter = (far + frr) / 2
        return tn, fp, fn, tp

    elif classification_method == "LogReg":  # This is an example of how can you use Logistic regression
        param_grid = [
            {'solver': ['newton-cg'],
             'C': [0.1, 0.2, 0.4, 0.45, 0.5],
             'penalty': ['l1', 'l2']}]  # Trying to improve

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        CUAuthModel = linear_model.LogisticRegression(random_state=random_seed, tol=1e-5)
        scoring_function = 'f1'  # You can use scoring function as HTER, see Aux_codes.py for details
        # Grid search for best parameter search .. using 10 fold cross validation and 'f1' as a scoring function.
        SearchTheBestParam = GridSearchCV(estimator=CUAuthModel, param_grid=param_grid, cv=10,
                                          scoring=scoring_function)
        SearchTheBestParam.fit(training_X, training_y)
        solver = SearchTheBestParam.best_params_['solver']
        cval = SearchTheBestParam.best_params_['C']
        penalty = SearchTheBestParam.best_params_['penalty']

        # Retraining the model again using the best parameter and testing, remember k = 1 will always give 100% accuracy :) on training data
        FinalModel = linear_model.LogisticRegression(solver=solver, C=cval, penalty=penalty,
                                                     random_state=random_seed, tol=1e-5)
        FinalModel.fit(training_X, training_y)
        pred_gen_lables = FinalModel.predict(genuine_testing)
        pred_imp_lables = FinalModel.predict(impostor_testing)
        pred_testing_labels = np.concatenate((pred_gen_lables, pred_imp_lables))

        # computing the error rates for the current predictions
        tn, fp, fn, tp = confusion_matrix(testing_y, pred_testing_labels).ravel()

        return tn, fp, fn, tp

    elif classification_method == "SVM":
        # running two class SVM
        clf = svm.SVC()

        #tuning SVM parameters
        param_grid = {'C': [0.1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
        SearchTheBestParam = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, scoring='accuracy')

        # Fitting the training data
        SearchTheBestParam.fit(training_X, training_y)

        #predicting the labels for the test data
        pred_gen_lables = SearchTheBestParam.predict(genuine_testing)
        pred_imp_lables = SearchTheBestParam.predict(impostor_testing)
        pred_testing_labels = np.concatenate((pred_gen_lables, pred_imp_lables))

        # computing the error rates for the current predictions
        tn, fp, fn, tp = confusion_matrix(testing_y, pred_testing_labels).ravel()
        return tn, fp, fn, tp
    elif classification_method == "RanFor":
        clf = RandomForestClassifier()
        # Number of features to consider at every split
        max_features = ['sqrt']
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]

        #Running cross validation to find the best parameters for classification method
        param_grid = {'n_estimators': [20, 50, 100],
                      'max_features': max_features,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
        SearchTheBestParam = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, scoring='accuracy')

        SearchTheBestParam.fit(training_X, training_y)

        #Predicting the labels of the testing data
        pred_gen_lables = SearchTheBestParam.predict(genuine_testing)
        pred_imp_lables = SearchTheBestParam.predict(impostor_testing)
        pred_testing_labels = np.concatenate((pred_gen_lables, pred_imp_lables))

        # computing the error rates for the current predictions
        tn, fp, fn, tp = confusion_matrix(testing_y, pred_testing_labels).ravel()

        return tn, fp, fn, tp


    else:  # Add more classification methods same as above
        raise ValueError('classification method unknown!')

def performance_metrics(tn, fp, fn, tp):
    """
    Given confusion matrix, function returns FAR, FRR and HTER
    """
    far = fp / (fp + tn) # computing FAR
    frr = fn / (fn + tp) # computing FRR
    hter = (far + frr) / 2 # computing HTER
    return far,frr,hter

def analyse_results(users,fars,frrs,hters):
    """
    Function to analyse results by reporting FAR, FRR and HTER
    """

    x = np.arange(len(users))
    ax1 = plt.subplot(1, 1, 1)
    w = 0.1
    plt.xticks(x + w / 2, users, rotation='vertical') # Adding x axis

    # making bar graph of FAR, FRR and HTER for each user
    farsP = ax1.bar(x, fars, width=w, color='b')
    frrsP = ax1.bar(x + w, frrs, width=w, color='g')
    htersP = ax1.bar(x + 2 * w, hters, width=w, color='r')
    plt.ylabel('Scores')
    ax1.legend(labels=['FAR', 'FRR', 'HTER'])
    plt.show()



# Generating label column for the given data matrix
def get_labels(data_matrix, label):
    if data_matrix.shape[0] > 1:
        label_column = np.empty(data_matrix.shape[0])
        label_column.fill(label)
    else:
        print('Warning! user data contains only one sample')
    return label_column


if __name__ == "__main__":

    final_result = pd.DataFrame(columns=['user', 'mode', 'method', 'tn', 'fp', 'fn', 'tp'])
    row_counter = 0
    #classification_method = ['SVM', 'kNN', 'LogReg', 'RanFor']
    classification_method = ['kNN']
    #phone_usage_mode = ['Landscape', 'Portrait']
    phone_usage_mode = ['Landscape']

    user_list = range(139, 144)  # Get the user list

    feature_selection_threshold = 20  # Try different numbers of features
    SMOTE_k = 7
    user_models = {}  # this is a dictionary of models for eac


    #Saving results of FAR FRR HTER in arrays to make bar graphs
    users = []
    fars = []
    frrs = []
    hters = []
    # Build biometric models for each user
    # There would two models for each user, namely, landscape and portrait

    for user in user_list:
        for mode in phone_usage_mode:
            # Training data
            training_file =  mode + 'Session1.csv'

            # Testing data
            testing_file = mode + 'Session2.csv'

            # Preprocess data
            training_data = preprocess(training_file)
            testing_data = preprocess(testing_file)

            # Extract features for training data
            for d in training_data:
                extract_features(d)

            # Extract features for testing data
            for d in testing_data:
                extract_features(d)

            #Running through each classification method
            for method in classification_method:
                # ----- training_X: this is training feature matrix that contains feature vectors from genuine and impostors

                training_X_gen = []  # genuine training data
                training_X_imp = []  # impostor training data
                for d in training_data:
                    if d.ID != user:

                        training_X_imp.append(d.features)
                    else:
                        training_X_gen.append(d.features)

                training_X = training_X_gen + training_X_imp  # training data

                # The genuine feature vectors are created from the data of genuine user collected in Session1
                # Similarly the impostor feature vectors are created from the data of users other than the genuine user collected in Session1
                # training_y: this is basically a column of ONES (genuine) and ZEROES (impostors)
                training_y_gen = get_labels(np.array(training_X_gen), 1)  # label for genuine data
                training_y_imp = get_labels(np.array(training_X_imp), 0)  # label for impostor data
                training_y = np.append(training_y_gen, training_y_imp)  # combining impostor and genuine

                # genuine_testing: feature matrix that consists of feature vectors created from the data of genuine user collected in Session2
                # impostor_testing: feature matrix that consists of feature vectors created from the data of users other than the genuine user collected in Session2

                genuine_testing = []  # genuine training data
                impostor_testing = []  # impostor training data
                for d in testing_data:
                    if d.ID != user:
                        impostor_testing.append(d.features)
                    else:
                        genuine_testing.append(d.features)


                # Select the features before running the classification, for example using mutual information
                training_X, training_y, genuine_testing, impostor_testing = select_features(training_X, training_y,
                                                                                            genuine_testing,
                                                                                            impostor_testing,
                                                                                            feature_selection_threshold)

                # Find the error rate using a classification method and save the errors

                tn, fp, fn, tp = get_error_rates(training_X, training_y, genuine_testing, impostor_testing, method)

                #using arrays to store FAR FRR HTER to make bar plots
                users.append(user)
                far,frr,hter = performance_metrics(tn, fp, fn, tp)
                fars.append(far)
                frrs.append(frr)
                hters.append(hter)

                # using panda dataframe to store all the results
                final_result.loc[row_counter] = [user, mode, method, tn, fp, fn, tp]
                row_counter = row_counter + 1

    # make bar plots of results for each user
    #--Code takes forever to run because data is very big and machine learning algorithms take up a lot of energy that our computer can't handle to run everythin
    analyse_results(users,fars,frrs,hters)
