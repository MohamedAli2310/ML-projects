#######################################################################################################################
import statistics

import numpy as np
import pandas as pd
from dtw import *
from scipy import signal
from scipy import interpolate
import itertools as it
from matplotlib import pyplot as plt
import os
from pathlib import Path
from scipy.spatial import distance



class MyScaler:
    # initializes a MyScalar object and takes the method in the constructor, with standard being the default
    def __init__(self, vector, method='standard'):
        self.vector = vector
        self.method = method
        # if standard is the method, we define mean and std
        if method == 'standard':
            self.mean = np.mean(vector)
            self.std = np.std(vector)
        # if minmax is the method, we define min and max
        elif method == 'minmax':
            self.min = np.min(vector)
            self.max = np.max(vector)
        else:
            raise ValueError('Unknown scaling method!')

    def get_scaling_params(self):
        # Based on the method, it returns a pair either (min, max) or (mean, std)
        if self.method == 'standard':
            return (self.mean, self.std)
        else:
            return (self.min, self.max)


    def transform(self, other):
        # Based on which method (standard or minmax) to be used, this function scales the other (a vector)
        # and returns the scaled vector
        if self.method == 'standard':
            # an array to store the normalized vector
            new_other = np.ndarray((other.size,))

            for i in range(other.size):
                new_other[i] = (other[i]-self.mean)/self.std
            return new_other

        else:
            # creates an array to store both vector and other in order to find the min and max of both arrays together
            concatenated = np.append(self.vector,other)
            new_other = np.ndarray((other.size,))
            for i in range (other.size):
                # creates the new_other array from the normalized entries based on universal min and max
                new_other[i]=(concatenated[self.vector.size +i]-np.min(concatenated))/(np.max(concatenated)-np.min(concatenated))
            return new_other


class Preprocessor:

    def __init__(self, signal):
        # signal is a one dimensional array, a time series, for example acceleration in dimension x
        self.signal = signal

    def remove_noise(self):
        self.signal = self.signal.rolling(window=3, min_periods=1).mean()

    def extract_KHT_and_KITs(self, df, mode):
        key_hold_dict = {}
        temporal_features_dict = {}
        if mode == "train":
            x = 0
            y = len(df)//2
        elif mode == "validate":
            x = len(df)//2
            y = len(df)*65//100
        elif mode == "test":
            x = len(df)*65//100
            y = len(df)-3
        else:
            x = df.iloc[0,0]
            y = df.iloc[len(df)-1,0]-3

        for i in range(x,y):
        # if (df.loc[i, "key"] in unigraphs):
            k_press = df.loc[i,'time']
            k_release = df.loc[i+1,'time']
            if (df.loc[i,"direction"] == 0 and df.loc[i+1,"direction"] == 1 and df.loc[i, "key"] == df.loc[i+1, "key"] ):
                if (not key_hold_dict.get(df.loc[i,"key"])):
                    key_hold_dict[df.loc[i,"key"]]=[]
                key_hold_dict[df.loc[i,"key"]].append(k_release - k_press)
        # if ((df.loc[i,"key"],df.loc[i+2,"key"]) in digraphs):
            k1_press = df.loc[i,"time"]
            k1_release = df.loc[i+1,"time"]
            k2_press = df.loc[i+2,"time"]
            k2_release = df.loc[i+3,"time"]
            if ((df.loc[i,"direction"] == 0 and df.loc[i+1,"direction"] == 1 and df.loc[i+2,"direction"] == 0 and
                    df.loc[i+3,"direction"] == 1) and (df.loc[i, "key"] == df.loc[i+1, "key"]) and
                    df.loc[i+2, "key"] == df.loc[i+3, "key"]):
                if (not temporal_features_dict.get(df.loc[i, "key"] + df.loc[i + 2, "key"] + '1')):
                    temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '1']=[]
                    temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '2']=[]
                    temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '3']=[]
                    temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '4']=[]
                temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '1'].append(k2_press-k1_release)
                temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '2'].append(k2_release-k1_release)
                temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '3'].append(k2_press-k1_press)
                temporal_features_dict[df.loc[i, "key"] + df.loc[i + 2, "key"] + '4'].append(k2_release-k1_press)
        return ((key_hold_dict,temporal_features_dict))

    def get_segments(self, df, window_length, sampling_rate):
        # size of one segment
        window_segment_size = window_length * sampling_rate
        # number of windows in the signal
        multiple = len(df) // (window_segment_size // 2)
        # signal trimmed to exactly match the number of windows
        trimmed_signal = df.iloc[:(window_segment_size // 2 * multiple)]
        numberOfWindows = multiple - 1
        # create a double array for the segments
        final_segment_array = np.ndarray(shape=(numberOfWindows, 1), dtype=object)
        for i in range(numberOfWindows):
            # starts at multiples of 125 (0,125,250,...) and ends after adding 250 points
            start = i*(window_segment_size // 2)
            end = start + window_segment_size + 1
            # convert the segment to a numpy array and add it to the array of segments
            final_segment_array[i][0] = trimmed_signal.iloc[start:end]
            # plt.plot(final_segment_array[i][0])
        # plt.show()
        # print(segment_array)
        return final_segment_array


class Comparators:

    def get_score(self, test, train_dict):
        distance_array = []
        for data in test:
            # print(type(data), data)
            test_dict = Preprocessor.extract_KHT_and_KITs(Preprocessor, data, "none")[0]
            for x,y in test_dict.items():
                test = np.asarray(test_dict.get(x))
                if (not train_dict.get(x)):
                    pass
                else:
                    train = np.asarray(train_dict.get(x))
                    # print(x, test, train)
                    if (test.size != train.size):
                        bigger_array = train if train.size >= test.size else test
                        smaller_array = test if test.size < train.size else train
                        x = np.arange(0, smaller_array.size)
                        y = smaller_array
                        if (y.size>1 and x.size>1):

                            f = interpolate.interp1d(x, y)
                            # creating y-new based on f, the relationship between template and query
                            ynew = f(np.linspace(0, smaller_array.size - 1, num=bigger_array.size, endpoint=True))

                            # using dtw-python package with keep_internals=True and distance_only=True to make it faster
                            # d = dtw(ynew, bigger_array, keep_internals=True, distance_only=True)
                            # print("DIST",np.linalg.norm(bigger_array - ynew))
                            # print (bigger_array.shape, ynew.shape)
                            # print(np.matrix.transpose(np.atleast_2d(bigger_array)))
                            distance_array.append(np.linalg.norm(bigger_array-ynew))
                            # distance_array.append(distance.mahalanobis(bigger_array, ynew,
                            #     numpy.linalg.inv(np.cov(np.matrix.transpose(np.atleast_2d(bigger_array))))))
                        else:
                            pass
                    else:
                        # d = dtw(train, test, keep_internals=True, distance_only=True)
                        # print("DIST", np.linalg.norm(train - test))
                        distance_array.append(np.linalg.norm(train - test))
                        # print("failed here")
                        # distance_array.append(distance.mahalanobis(train, test, np.cov(train,test)))
        return statistics.median(distance_array)

#######################################################################################################################
class Evaluator:
    def __init__(self, gen_scores, imp_scores):
        self.gen_scores = gen_scores
        self.imp_scores = imp_scores
        self.thresholds = np.linspace(start=0, stop=1, num=100, endpoint=True)
        # print("GEN AND IMP",gen_scores, imp_scores)
    # SAME AS LAB 1
    def get_fars(self):
        fars_array = np.zeros((self.thresholds.size, ))
        for i in range(self.thresholds.size):
            fars_count = 0
            for score in self.imp_scores:
                if score >= self.thresholds[i]:
                    fars_count += 1
            fars_array[i] = fars_count / self.imp_scores.size
            # returns fars
        return fars_array

    # SAME AS LAB 1
    def get_frrs(self):
        frrs_array = np.zeros((self.thresholds.size,))
        for i in range(self.thresholds.size):
            frrs_count = 0
            for score in self.gen_scores:
                if score < self.thresholds[i]:
                    frrs_count += 1
            frrs_array[i] = frrs_count / self.gen_scores.size
            # returns fars
        return frrs_array

    # SAME AS LAB 1
    def plot_roc(self):
        # performance values and then plotting them out with pyplot
        tar_array = np.zeros((self.thresholds.size, ))
        far_array = self.get_fars()
        frr_array = self.get_frrs()
        for i in range(far_array.size):
            tar_array[i] = 1 - frr_array[i]
        for i in range(self.gen_scores.size):
            plt.plot(self.get_fars(), tar_array, 'ro--', linewidth=2, markersize=6)
        plt.title('ROC Plot')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Accept Rate')
        plt.xlabel('False Accept Rate')
        plt.show()

    # SAME AS LAB 1
    def plot_det(self):
        for i in range(self.gen_scores.size):
            plt.plot(self.get_frrs(), self.get_fars(), 'ro--', linewidth=2, markersize=6)
            idx = np.argwhere(np.diff(np.sign(self.get_fars() - self.get_frrs()))).flatten()
            idx = idx[1] if idx.size > 1 else idx
            # plot out the point of the intersect (EER) on a separate red graph
            plt.plot(self.get_frrs()[idx], self.get_frrs()[idx], 'go')
        plt.title('DET Plot')
        plt.plot(self.get_frrs(), self.get_frrs(), 'g--', linewidth=2)
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('False Accept Rate')
        plt.xlabel('False Reject Rate')

        plt.show()

    # SAME AS LAB 1
    def get_eer(self):
        # Use the implementation of this function from Lab1
        idx = np.argwhere(np.diff(np.sign(self.get_fars() - self.get_frrs()))).flatten()
        return self.get_frrs()[idx][0]

    def get_eer_threshold(self):
        eer = self.get_eer()
        fars_array = self.get_fars()
        # np.where finds the index where far = eer, this is the required threshold
        index = np.where(fars_array == eer)
        return self.thresholds[index]

#######################################################################################################################
class Person:
    def __init__(self, user_name, gen_train, gen_validate, gen_test, imp_test):
        self.user_name = user_name
        self.gen_train = gen_train
        self.gen_validate = gen_validate
        self.gen_test = gen_test
        self.imp_test = imp_test.squeeze()
        self.gen_scores
        self.imp_scores
        self.heat_map_scores = {}

    def extract_raw_data(self, data_matrix):
        final_matrix = np.ndarray((data_matrix.shape[0], len(data_matrix[data_matrix.shape[0] - 1][0])))
        for i in range(data_matrix.shape[0]):
            for j in range(len(data_matrix[data_matrix.shape[0] - 1][0])):
                final_matrix[i][j] = data_matrix[i][0][j]
        return final_matrix

        # uses scalers to scale a data matrix
        def scale_matrix_based_on_gen_train(self, scalers, data_matrix):
            for i in range(len(scalers)):
                # transform the column of the matrix based on scalar i
                transformed_column = scalers[i].transform(data_matrix[:, i])
                for k in range(transformed_column.shape[0]):
                    data_matrix[k][i] = transformed_column[k]
            return data_matrix

        # a method to make the rows of a given mamtrix of the same size
        def equalize_matrix_rows(self, data_matrix):
            max_length = [0]
            for data in data_matrix:
                if data[0].size > max_length[0]:
                    max_length[0] = data[0].size

            final_matrix = np.zeros((data_matrix.shape[0], max_length[0]))
            for i in range(data_matrix.shape[0]):
                # interpolates the rows of sizes smaller than max size to reach max size
                final_matrix[i] = self.interpolate_matrix_rows(data_matrix[i][0], max_length[0])
            return final_matrix

        # helper method for equalize_matrix_rows. Used to interpolate a row to a specific length
        def interpolate_matrix_rows(self, row, length):
            if (row.size != length):
                # creates an array for the x-axis (0,1,2,....,smaller_array.size)
                x = np.arange(0, row.size)
                y = row
                # extracts the relationship between x and the smaller array
                f = interpolate.interp1d(x, y)
                # creates a new interpolated array of size = bigger_array.size from the smaller array.
                ynew = f(np.linspace(0, row.size - 1, num=length, endpoint=True))
                return ynew
            else:
                return row

        def set_math_scores(self):

            # preprocess gen_train, removes noise, and get segments
            preprocessor = Preprocessor(self.gen_train)
            preprocessor.remove_noise()
            processed_gen_train = preprocessor.get_segments()
            # preprocess gen-test, removes noise, and get segments
            preprocessor = Preprocessor(self.gen_test)
            preprocessor.remove_noise()
            processed_gen_test = preprocessor.get_segments()
            # preprocess imp-test, removes noise, and get segments
            preprocessor = Preprocessor(self.imp_test)
            preprocessor.remove_noise()
            processed_imp_test = preprocessor.get_segments()

            (training_key_hold_dict, training_temporal_features_dict) = Preprocessor.extract_KHT_and_KITs(Preprocessor,
                                                                                                          df, "train")
            validate_segments = Preprocessor.get_segments(Preprocessor, gen_validate, 10, 30)
            test_segments = Preprocessor.get_segments(Preprocessor, gen_test, 10, 30)
            gen_train_data = self.equalize_matrix_rows(processed_gen_train)
            gen_test_data = self.equalize_matrix_rows(processed_gen_test)
            imp_test_data = self.equalize_matrix_rows(processed_imp_test)
            # Creates a Comparator with gen train data
            comparator = Comparators(gen_train_data)
            gen_scores = np.zeros((gen_test_data.shape[0],))
            imp_scores = np.zeros((imp_test_data.shape[0],))
            # compares all gen scores with the train data and get scores
            for i in range(gen_test_data.shape[0]):
                gen_scores[i] = comparator.get_scores(gen_test_data[i])
            # compares all imp scores with the train data and get scores
            for i in range(imp_test_data.shape[0]):
                imp_scores[i] = comparator.get_scores(imp_test_data[i])
            # print(Comparators.get_score(Comparators, validate_segments[0], training_key_hold_dict))
            # print(Comparators.get_score(Comparators, test_segments[0], training_key_hold_dict))
            #
            # gen_train_features = np.zeros((processed_gen_train.size, 10))
            # gen_test_features = np.zeros((processed_gen_test.size, 10))
            # imp_test_features = np.zeros((processed_imp_test.size, 10))
            # features = [gen_train_features, gen_test_features, imp_test_features]
            # processed_data = [processed_gen_train, processed_gen_test, processed_imp_test]
            # gen_train_scalers = []
            #
            #
            # gen_train_data = self.equalize_matrix_rows(processed_gen_train)
            # gen_test_data = self.equalize_matrix_rows(processed_gen_test)
            # imp_test_data = self.equalize_matrix_rows(processed_imp_test)
            #
            # # Creates a Comparator with gen train data
            # comparator = Comparators(gen_train_data)
            # gen_scores = np.zeros((gen_test_data.shape[0],))
            # imp_scores = np.zeros((imp_test_data.shape[0],))
            # # compares all gen scores with the train data and get scores
            # for i in range(gen_test_data.shape[0]):
            #     gen_scores[i] = comparator.get_scores(gen_test_data[i])
            # # compares all imp scores with the train data and get scores
            # for i in range(imp_test_data.shape[0]):
            #     imp_scores[i] = comparator.get_scores(imp_test_data[i])
            #     # prints imp and gen final median scores
            # print("IMP SCORES", np.median(imp_scores))
            # print("GEN SCORES", np.median(gen_scores))

            self.gen_scores = gen_scores
            self.imp_scores = imp_scores



#####################################################################################
class ResultAnalysis:
    def __init__(self, Population):
        self.Population = Population

        # A method to normalize gen and imp scores based on the comparison scheme
        def normalize_gen_imp_scores(self, og_gen_scores, og_imp_scores, comparison_scheme):

            # defines scalars for gen and imp
            gen_score_scaler = MyScaler(og_gen_scores, method="minmax")
            imp_score_scaler = MyScaler(og_imp_scores, method="minmax")
            # scales imp based on gen
            imp_scores = gen_score_scaler.transform(og_imp_scores)
            # scales gen based on imp
            gen_scores = imp_score_scaler.transform(og_gen_scores)
            processed_gen_scores = np.zeros(gen_scores.shape)
            # using 1 - genscores to fix the problem where histogram data are the further the better where
            # dtw and pointwise are the opposite
            for i in range(gen_scores.shape[0]):
                processed_gen_scores[i] = 1 - gen_scores[i]
            processed_imp_scores = np.zeros(imp_scores.shape)
            # using 1 - impscores to fix the problem where histogram data are the further the better where
            # dtw and pointwise are the opposite
            for i in range(imp_scores.shape[0]):
                processed_imp_scores[i] = 1 - imp_scores[i]
            processed_gen_scores = np.sort(processed_gen_scores)
            processed_imp_scores = np.sort(processed_imp_scores)
            return (processed_gen_scores, processed_imp_scores)

    def analyze(self):
        for person in self.Population:
            processed_scores = self.normalize_gen_imp_scores(person.gen_scores, person.imp_scores)
            evaluator = Evaluator(processed_scores[0], processed_scores[1])
            evaluator.plot_roc()
            evaluator.plot_det()
            evaluator.get_eer()


if __name__ == "__main__":


    training_key_hold_dict={}
    training_temporal_features_dict={}
    validate_key_hold_dict={}
    validate_temporal_features_dict={}
    test_key_hold_dict={}
    test_temporal_features_dict={}


    devices = ["Desktop", "Tablet", "Phone"]
    for device in devices:
        data_folder_name = device
        data_path = os.path.join(Path(__file__).parents[0], 'RawKeystrokeData', data_folder_name)
        user_list = os.listdir(data_path)
        user_list = [user_list[0]]  # working on one user

        for user in user_list:
            print(f'working on {user}, {device}')



            # Training dataframe
            df = pd.read_csv(os.path.join(data_path, user))
            validate_slice = df.iloc[len(df)//2: len(df)*65//100]
            test_slice = df.iloc[len(df)*65//100: len(df)]

            (training_key_hold_dict, training_temporal_features_dict) = Preprocessor.extract_KHT_and_KITs(Preprocessor, df, "train")
            validate_segments = Preprocessor.get_segments(Preprocessor, validate_slice, 10, 30)
            test_segments = Preprocessor.get_segments(Preprocessor, test_slice, 10, 30)
            print(Comparators.get_score(Comparators, validate_segments[0], training_key_hold_dict))
            print(Comparators.get_score(Comparators, test_segments[0], training_key_hold_dict))

            # for data in validate_sliced:
            #     validate_key_hold_dict.update(preprocessor.extract_KHT_and_KITs(preprocessor, data, "none")[0])
            #     validate_temporal_features_dict.update(preprocessor.extract_KHT_and_KITs(preprocessor, data, "none")[1])
            # (test_key_hold_dict, test_temporal_features_dict) = preprocessor.extract_KHT_and_KITs(preprocessor, df, "test")

    # training_key_hold_dict = np.asarray(training_key_hold_dict)



    # for x, y in validate_key_hold_dict.items():
    #     print(x,type(y),y)
    # print("AAAAAAAA&&&&&&&&&&&AAAAAAAAAAa")
    # for x, y in training_temporal_features_dict.items():
    #     print(x, y)