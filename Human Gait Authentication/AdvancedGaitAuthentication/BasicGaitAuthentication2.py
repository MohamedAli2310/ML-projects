# Install packages as per the need
import numpy as np
import pandas as pd
from dtw import *
from scipy import signal
from scipy import interpolate
import itertools as it
from matplotlib import pyplot as plt
import seaborn as sns
import os
from pathlib import Path

#######################################################################################################################
class Params:
    # This class has most of the parameter that we will be using in this project
    # Feel free to define or add you own parameter, if you think that will help you
    # Keep your code clean
    column_names = ['x', 'y', 'z', 'timestamps']  # defining the name of the columns
    sensor_file_list = ['Accelerometer_Data.txt']  # We will be using only accelerometer data
    # Initializing some constants:
    global_threshold = 0.5  # Setting this as 0.5 assuming that your scores shall be normalized between 0 to 1
    random_seed = 1833  # Just a seed value so the exp are reproducible, think why 1833 as a seed?
    sampling_rate = 25  # for simplicity we using the fixed sampling rate which is mostly correct
    window_length = 10  # seconds
    sliding_interval = int(window_length / 2)  # no need of this variable we will be using window_length/2 as as rule
    # in the future we may use more than one sensors thus the list instead of a variable
    # May need in the future
    segmentation_schemes = ['cycle']
    comparison_schemes = ['pointwise', 'dtw', 'hist']
    comparison_space = ['statistical', 'frequency', 'stat-freq']
    num_bins = 100

#######################################################################################################################
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

    # inverts the things done by transform
    def inverse_transform(self, other):
        if self.method == 'standard':
            new_other = np.ndarray((other.size,))
            for i in range(other.size):
                new_other[i]= (other[i]*self.std) + self.mean
            return new_other
        else:
            new_other = np.ndarray((other.size,))
            for i in range(other.size):
                new_other[i] = (other[i] * (self.max-self.min)) + self.min
            return new_other

#######################################################################################################################
class Preprocessor:
    def __init__(self, signal):
        # signal is a one dimensional array, a time series, for example acceleration in dimension x
        self.signal = signal

    def remove_noise(self):
        self.signal = self.signal.rolling(window=3, min_periods=1).mean()

    def get_segments(self, segmentation_scheme):
        if segmentation_scheme == 'sliding_window':
            Params.window_length = 10
            # size of one segment
            window_segment_size = Params.window_length * Params.sampling_rate
            # number of windows in the signal
            multiple = len(self.signal) // (window_segment_size // 2)
            # signal trimmed to exactly match the number of windows
            trimmed_signal = self.signal[:(window_segment_size // 2 * multiple)]
            numberOfWindows = multiple - 1
            # create a double array for the segments
            final_segment_array = np.ndarray(shape=(numberOfWindows, 1), dtype=object)
            for i in range(numberOfWindows):
                # starts at multiples of 125 (0,125,250,...) and ends after adding 250 points
                start = i*(window_segment_size // 2)
                end = start + window_segment_size + 1
                # convert the segment to a numpy array and add it to the array of segments
                final_segment_array[i][0] = trimmed_signal.to_numpy()[start:end]
                # plt.plot(final_segment_array[i][0])
            # plt.show()
            # print(segment_array)
            return final_segment_array

        elif segmentation_scheme == 'cycle':
            Params.window_length = 2
            window_segment_size = Params.window_length * Params.sampling_rate
            multiple = len(self.signal) // (window_segment_size // 2)
            trimmed_signal = self.signal[:(window_segment_size//2 * multiple)]
            numberOfWindows = multiple - 1
            segment_array = np.ndarray((numberOfWindows, 1), dtype=object)
            final_segment_array = np.ndarray(shape=(numberOfWindows, 1), dtype=object)
            for i in range(numberOfWindows):
                start = i * (window_segment_size // 2)
                end = start + window_segment_size + 1
                # print(segment_array[i+1])
                # print(self.signal['x'].to_numpy()[start:end])
                segment_array[i][0] = trimmed_signal.to_numpy()[start:end]
            # print("segment_array", segment_array[0])
            for i in range(segment_array.size):
                segment = segment_array[i][0]
                # used find_peaks to find local maxima. we set the distance parameter to 6 to prevent having 2 maxima
                # very close to each other (at the same hump)
                peaks = signal.find_peaks(segment, distance=6)
                # cuts the segment from one maximum to the other to produce a cycle
                final_segment_array[i][0] = segment[peaks[0][-2]:peaks[0][-1]+1]
                # plt.plot(final_segment_array[i][0])
            # plt.show()
            return final_segment_array
        else:
            raise ValueError('Unknown segmentation_scheme!')

#######################################################################################################################
class Comparators:
    def __init__(self, template):
        # Template and query as 2D numpy array
        self.template = template  # gen_train 2D matrix
        self.query = None  # one of the rows from either gen_test or imp_test 2D matrices

    def get_scores(self, query, comparison_scheme='dtw'):
        self.query = query
        if comparison_scheme == 'pointwise':
            # use the pointwise_distance() to get the scores
            return self.pointwise_distance()
        elif comparison_scheme == 'dtw':
            # use the dtw_distance() function to get the scores
            return self.dtw_distance()
        elif comparison_scheme == 'hist':
            # use the hist_distance() function to get the scores
            return self.hist_distance()
        else:
            raise ValueError('Unknown comparison_scheme!')

    def pointwise_distance(self):
        distance_array = np.zeros((self.template.shape[0], ))
        for i in range(self.template.shape[0]):
            template = self.template[i]
            # determines whether template or query is bigger
            if (template.size != self.query.size):
                bigger_array = template if template[0].size >= self.query.size else self.query
                smaller_array = template if template[0].size < self.query.size else self.query
                # creates an array for the x-axis (0,1,2,....,smaller_array.size)
                x = np.arange(0, smaller_array.size)
                y = smaller_array
                # extracts the relationship between x and the smaller array
                f = interpolate.interp1d(x, y)
                # creates a new interpolated array of size = bigger_array.size from the smaller array.
                ynew = f(np.linspace(0, smaller_array.size-1, num=bigger_array.size, endpoint=True))
                # finds the distance between the two vectors using linalg.norm
                distance_array[i] = np.linalg.norm(bigger_array - ynew)
            else:
                distance_array[i] = np.linalg.norm(template - self.query)
        # return the median of the distances as the final match score
        return np.median(distance_array)

    def dtw_distance(self):

        distance_array = np.zeros((self.template.shape[0], ))
        i = 0
        for template in self.template:
            if (template.size != self.query.size):
                bigger_array = template if template.size >= self.query.size else self.query
                smaller_array = template if template.size < self.query.size else self.query
                x = np.arange(0, smaller_array.size)
                y = smaller_array
                f = interpolate.interp1d(x, y)
                # creating y-new based on f, the relationship between template and query
                ynew = f(np.linspace(0, smaller_array.size - 1, num=bigger_array.size, endpoint=True))

                # using dtw-python package with keep_internals=True and distance_only=True to make it faster
                d = dtw(ynew, bigger_array, keep_internals=True, distance_only=True)
                distance_array[i] = d.distance
                i += 1
            else:
                d = dtw(self.query, template, keep_internals=True, distance_only=True)
                distance_array[i] = d.distance
                i += 1
        return np.median(distance_array)

    def hist_distance(self):

        distance_array = np.zeros((self.template.shape[0], ))
        i = 0
        for template in self.template:
            if (template.size != self.query.size):
                bigger_array = template if template.size >= self.query.size else self.query
                smaller_array = template if template.size < self.query.size else self.query
                x = np.arange(0, smaller_array.size)
                y = smaller_array
                f = interpolate.interp1d(x, y)
                ynew = f(np.linspace(0, smaller_array.size - 1, num=bigger_array.size, endpoint=True))
                np.linspace(0, smaller_array.size, num=bigger_array.size)
                distance_array[i] = self.hist_distance_helper(bigger_array, ynew)
            else:
                distance_array[i] = self.hist_distance_helper(self.query, template)
            i += 1

        return np.median(distance_array)

    def hist_distance_helper(self, template_segment, query_segment):
        # This would basically contain the same code that you implemented in Lab1 to find the hist intersection
        # It is receiving two vectors and would return amount of intersection between them
        hist_1, _ = np.histogram(template_segment, bins=20, range=[template_segment.min(), template_segment.max()])
        hist_2, _ = np.histogram(query_segment, bins=20, range=[query_segment.min(),query_segment.max()])
        minima = np.minimum(hist_1, hist_2)
        # the intersection can be thought of as the percentage of overlap by the imposter onto the genuine scores \
        # over the totaL imposter scores.
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        # intersection is returned as a fraction/decimal to determine how good or bad a biometric system is with respect to
        # the set of probabilistic scores of its imposters
        return intersection

    # *************************************** Following implementations are for Lab3 ***************************
    # def feature_level_distance_timedomain(self):
    # More description for this method shall be provided next week
    #     # Find match scores and return
    #     pass

    # def feature_level_distance_frequencydomain(self):
    # More description for this method shall be provided next week
    #     # Find match scores and return
    #     pass

    # def feature_level_distance_cross_correlation(self):
    # More description for this method shall be provided next week
    #     pass

    # def your_own_distance(self): Bonus: 10 points!!
    # Can you think of a new distance measure that would beat the rest :) [Without using machine learning]
    # Make sure your describe this method in the report properly.
    #     # Find match scores and return
    #     pass


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

    def get_hists_overlap(self):
        # converts gen_and imp scores into histograms (hist_1 and hist_2) before finding minima, the minimum between the two
        # histograms.
        hist_1, _ = np.histogram(self.gen_scores, bins=100, range=[0, 1])
        hist_2, _ = np.histogram(self.imp_scores, bins=100, range=[0, 1])
        minima = np.minimum(hist_1, hist_2)
        # the intersection can be thought of as the percentage of overlap by the imposter onto the genuine scores
        # over the total imposter scores.
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        # plot out the histogram of all imposter scores, genuine scores and the overlap (done with a bar chart)
        plt.hist(self.imp_scores, bins=100, range=[0, 1])
        plt.hist(self.gen_scores, bins=100, range=[0, 1])
        plt.bar(np.arange(0, 1, 0.01), minima, 0.01, color="yellow")
        plt.show()
        # intersection is returned as a fraction/decimal to determine how good or bad a biometric system is with
        # respect to the set of probabilistic scores of its imposters
        return intersection

#######################################################################################################################
class Person:
    def __init__(self, user_name, gen_train, gen_test, imp_test):
        self.user_name = user_name
        self.gen_train = gen_train
        self.gen_test = gen_test
        self.imp_test = imp_test.squeeze()
        self.gen_scores = {}  # This is a dictionary the keys of which will be the combination of segmentation_scheme
        # and comparision_scheme separated by hyphen i.e. '-', For example, we compute the gen_scores using
        # segmentation_scheme 'cycle' and comparision_scheme 'dtw' then the entry to the dictionary will look like below
        # self.gen_scores[segmentation_scheme+'-'+comparision_scheme] = [the list of scores obtained from Comparators]
        # I hope you got that the values corresponding to each key is basically the list of scores obtained for that
        # key (combination of segmentation and comparision scheme
        self.imp_scores = {}  # Same as self.gen_scores
        self.heat_map_scores = {}

    # By default we have segmentation_scheme = 'cycle', comparision_scheme = 'dtw', once you pass the values
    # it the passed values will automatically replace the default values

    # a method to extract statistical features
    def extract_stats_features(self, data_matrix):
        # initializes a matrix to store the features
        final_matrix = np.ndarray((data_matrix.shape[0], 10))
        for j in range(data_matrix.shape[0]):
            final_matrix[j][0] = np.amin(data_matrix[j][0])
            final_matrix[j][1] = np.max(data_matrix[j][0])
            final_matrix[j][2] = np.mean(data_matrix[j][0])
            final_matrix[j][3] = np.std(data_matrix[j][0])
            final_matrix[j][4] = np.percentile(data_matrix[j][0], 25)
            final_matrix[j][5] = np.percentile(data_matrix[j][0], 50)
            final_matrix[j][6] = np.percentile(data_matrix[j][0], 75)
            final_matrix[j][7] = (scipy.signal.find_peaks(data_matrix[j][0])[0]).size
            final_matrix[j][8] = scipy.stats.kurtosis(data_matrix[j][0])
            final_matrix[j][9] = scipy.stats.skew(data_matrix[j][0])
        return final_matrix

    # a method to extract freq features
    def extract_freq_features(self, data_matrix):
        # initializes a matrix to store the features
        final_matrix = np.ndarray((data_matrix.shape[0], 10))
        for j in range(data_matrix.shape[0]):
            frequencies = np.fft.fft(data_matrix[j][0])
<<<<<<< HEAD
=======
            # just taking the first half of the fft features
>>>>>>> b616e787c421e690e884dc4b051df5d8b9729ae2
            frequencies = np.abs(frequencies)[:frequencies.size // 2] * (2 / frequencies.size)
            final_matrix[j][0] = np.min(frequencies)
            final_matrix[j][1] = np.max(frequencies)
            final_matrix[j][2] = np.percentile(frequencies, 5)
            final_matrix[j][3] = np.percentile(frequencies, 25)
            final_matrix[j][4] = np.percentile(frequencies, 50)
            final_matrix[j][5] = np.percentile(frequencies, 75)
            final_matrix[j][6] = np.mean(frequencies)
            final_matrix[j][7] = np.std(frequencies)
            final_matrix[j][8] = np.max(frequencies) / np.min(frequencies)
            final_matrix[j][9] = np.percentile(frequencies, 95)
        return final_matrix

    # a method to extract raw data to look the same as stats and freq
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
            final_matrix[i]= self.interpolate_matrix_rows(data_matrix[i][0], max_length[0])
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

    def set_math_scores(self, comp_space='raw_data', segmentation_scheme='cycle', comparision_scheme='dtw'):

        # preprocess gen_train, removes noise, and get segments using the specified comparison scheme
        preprocessor = Preprocessor(self.gen_train)
        preprocessor.remove_noise()
        print("gentrain", segmentation_scheme, comparision_scheme)
        processed_gen_train = preprocessor.get_segments(segmentation_scheme)
        # preprocess gen-test, removes noise, and get segments using the specified comparison scheme
        preprocessor = Preprocessor(self.gen_test)
        preprocessor.remove_noise()
        print("gentest", segmentation_scheme, comparision_scheme)
        processed_gen_test = preprocessor.get_segments(segmentation_scheme)
        # preprocess imp-test, removes noise, and get segments using the specified comparison scheme
        preprocessor = Preprocessor(self.imp_test)
        preprocessor.remove_noise()
        print("imptest", segmentation_scheme, comparision_scheme)
        processed_imp_test = preprocessor.get_segments(segmentation_scheme)

        gen_train_features = np.zeros((processed_gen_train.size, 10))
        gen_test_features = np.zeros((processed_gen_test.size, 10))
        imp_test_features = np.zeros((processed_imp_test.size, 10))
        features = [gen_train_features, gen_test_features, imp_test_features]
        processed_data = [processed_gen_train, processed_gen_test, processed_imp_test]
        gen_train_scalers = []

        # Preprocessing: This part is basically what you have already implemented in lab2
        # Step1: Use the Preprocessor class to preprocess the gen_train, gen_test, and imp_test signals separately
        # Use the remove_noise() function to remove noise and then use the get_segments() of class Preprocessor The
        # outcome of the get_segments() would be a collection of cycles or a collection of sliding windows....
        # Technically, it would be a 2D numpy array You will have 2D numpy array for each of the three signals (
        # gen_train, gen_test, and imp_test) say gen_train_segment, gen_test_segment, and imp_test_segments

        if comp_space == 'raw_data':  # Already implemented in lab2
            gen_train_data = self.equalize_matrix_rows(processed_gen_train)
            gen_test_data = self.equalize_matrix_rows(processed_gen_test)
            imp_test_data = self.equalize_matrix_rows(processed_imp_test)
            # gen_train_data = self.equalize_matrix_rows(gen_train_data) if (
            #             segmentation_scheme == "cycle") else gen_train_data
            # gen_test_data = self.equalize_matrix_rows(gen_test_data) if (
            #             segmentation_scheme == "cycle") else gen_test_data
            # imp_test_data = self.equalize_matrix_rows(imp_test_data) if (
            #             segmentation_scheme == "cycle") else imp_test_data

            # Creates a Comparator with gen train data
            comparator = Comparators(gen_train_data)
            gen_scores = np.zeros((gen_test_data.shape[0],))
            imp_scores = np.zeros((imp_test_data.shape[0],))
            # compares all gen scores with the train data and get scores
            for i in range(gen_test_data.shape[0]):
                gen_scores[i] = comparator.get_scores(gen_test_data[i], comparision_scheme)
            # compares all imp scores with the train data and get scores
            for i in range(imp_test_data.shape[0]):
                imp_scores[i] = comparator.get_scores(imp_test_data[i], comparision_scheme)
                # prints imp and gen final median scores
            print("IMP SCORES", np.median(imp_scores))
            print("GEN SCORES", np.median(gen_scores))

            self.gen_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = gen_scores
            self.imp_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = imp_scores
            # Step 2: Use the Comparators class to obtain the genuine and impostor scores Basically, you will initialize
            # an Object of the Comparator class with "gen_train". The gen_train is basically the template for that user
            # then you will call get_scores(query, comparision_scheme) function to get the score for the query
            # the query is basically the test vector (rows) of gen_test and imp_test
            # if the query is from gen_test then the score will be appended to gen_scores,
            # else if the query is from imp_test then the score will be appended to imp_scores,
            # Finally, self.gen_scores and self.imp_scores shall be updated with the gen_scores and imp_scores respectively

        elif comp_space == 'statistical': # Only sliding window
            # Extracting statistical methods for gen_train, gen_test, and imp_test
            gen_train_features = self.extract_stats_features(processed_gen_train)
            gen_test_features = self.extract_stats_features(processed_gen_test)
            imp_test_features = self.extract_stats_features(processed_imp_test)
            # creating scalars using gen_train data
            for i in range (gen_train_features.shape[1]):
                gen_train_scalers.append(MyScaler(gen_train_features[:,i]))
<<<<<<< HEAD

=======
            # Scaling based on gen_train scalars
>>>>>>> b616e787c421e690e884dc4b051df5d8b9729ae2
            gen_train_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, gen_train_features)
            gen_test_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, gen_test_features)
            imp_test_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, imp_test_features)

            # Creates a Comparator with gen train data
            comparator = Comparators(gen_train_features)
            gen_scores = np.zeros((gen_test_features.shape[0],))
            imp_scores = np.zeros((imp_test_features.shape[0],))
            # compares all gen scores with the train data and get scores
            for i in range(gen_test_features.shape[0]):
                gen_scores[i] = comparator.get_scores(gen_test_features[i], comparision_scheme)
            # compares all imp scores with the train data and get scores
            for i in range(imp_test_features.shape[0]):
                imp_scores[i] = comparator.get_scores(imp_test_features[i], comparision_scheme)
            # prints imp and gen final median scores
            print("IMP SCORES", comp_space, np.median(imp_scores))
            print("GEN SCORES", comp_space, np.median(gen_scores))
            self.gen_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = gen_scores
            self.imp_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = imp_scores
            # In this section you will extract the statistical features (at least 10, see the list on the lecture slides)
            # from gen_train, gen_test, and imp_test ==> Remember? all these are 2D matrices containing a list of windows
            # For each window, you will extract 10 features, that means you will get 3 new 2D matrices, say
            # gen_train_fmatrix, gen_test_fmatrix, and imp_test_fmatrix; the size of these matrices would be (number of windows, number of features)
            # The gen_train_fmatrix shall serve a the template
            # for the user. Next, you will call get_scores(query, comparision_scheme) function to get the score for query.
            # The query is basically the test vector (rows) of gen_test_fmatrix and imp_test_fmatrix
            # if the query is from gen_test_fmatrix then the score will be appended to gen_scores,
            # else if the query is from imp_test_fmatrix then the score will be appended to imp_scores,
            # Finally, self.gen_scores and self.imp_scores shall be updated with the gen_scores and imp_scores respectively

        elif comp_space == 'frequency': # Only sliding window

            gen_train_features = self.extract_freq_features(processed_gen_train)
            gen_test_features = self.extract_freq_features(processed_gen_test)
            imp_test_features = self.extract_freq_features(processed_imp_test)
<<<<<<< HEAD

=======
>>>>>>> b616e787c421e690e884dc4b051df5d8b9729ae2

            for i in range(gen_train_features.shape[1]):
                gen_train_scalers.append(MyScaler(gen_train_features[:, i]))


            gen_train_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, gen_train_features)
            gen_test_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, gen_test_features)
            imp_test_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, imp_test_features)


            # Creates a Comparator with gen train data
            comparator = Comparators(gen_train_features)
            gen_scores = np.ndarray((gen_test_features.shape[0],))
            imp_scores = np.ndarray((imp_test_features.shape[0],))
            # compares all gen scores with the train data and get scores
            for i in range(gen_test_features.shape[0]):
                gen_scores[i] = comparator.get_scores(gen_test_features[i], comparision_scheme)
            # compares all imp scores with the train data and get scores
            for i in range(imp_test_features.shape[0]):
                imp_scores[i] = comparator.get_scores(imp_test_features[i], comparision_scheme)
            # prints imp and gen final median scores
            print("IMP SCORES", comp_space, np.median(imp_scores))
            print("GEN SCORES", comp_space, np.median(gen_scores))
            self.gen_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = gen_scores
            self.imp_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = imp_scores
            # In this section you will extract the frequency features (at least 10, try to think what features makes more sense)
            # from gen_train, gen_test, and imp_test ==> Remember? all these are 2D matrices containing a list of windows
            # For each window, you will extract 10 features, that means you will get 3 new 2D matrices, say
            # gen_train_fmatrix, gen_test_fmatrix, and imp_test_fmatrix; the size of these matrices would be (number of windows, number of features)
            # The gen_train_fmatrix shall serve a the template
            # for the user. Next, you will call get_scores(query, comparision_scheme) function to get the score for query.
            # The query is basically the test vector (rows) of gen_test_fmatrix and imp_test_fmatrix
            # if the query is from gen_test_fmatrix then the score will be appended to gen_scores,
            # else if the query is from imp_test_fmatrix then the score will be appended to imp_scores,
            # Finally, self.gen_scores and self.imp_scores shall be updated with the gen_scores and imp_scores respectively

        elif comp_space == 'stat-freq': # Only sliding window
            # stacks stat and freq features in one np array
            gen_train_features = np.hstack((self.extract_stats_features(processed_gen_train), self.extract_freq_features(processed_gen_train)))
            gen_test_features = np.hstack((self.extract_stats_features(processed_gen_test), self.extract_freq_features(processed_gen_test)))
            imp_test_features = np.hstack((self.extract_stats_features(processed_imp_test), self.extract_freq_features(processed_imp_test)))
            for i in range(gen_train_features.shape[1]):
                gen_train_scalers.append(MyScaler(gen_train_features[:, i]))
            # for i in range(gen_train_features.shape[0]):
            #     plt.plot(np.arange(20), gen_train_features[i])
            # plt.show()
            gen_train_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, gen_train_features)
            gen_test_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, gen_test_features)
            imp_test_features = self.scale_matrix_based_on_gen_train(gen_train_scalers, imp_test_features)
            # Creates a Comparator with gen train data
            comparator = Comparators(gen_train_features)
            gen_scores = np.ndarray((gen_test_features.shape[0],))
            imp_scores = np.ndarray((imp_test_features.shape[0],))
            # compares all gen scores with the train data and get scores
            for i in range(gen_test_features.shape[0]):
                gen_scores[i] = comparator.get_scores(gen_test_features[i], comparision_scheme)
            # compares all imp scores with the train data and get scores
            for i in range(imp_test_features.shape[0]):
                imp_scores[i] = comparator.get_scores(imp_test_features[i], comparision_scheme)
            # prints imp and gen final median scores
            print("IMP SCORES", comp_space, np.median(imp_scores))
            print("GEN SCORES", comp_space, np.median(gen_scores))

            self.gen_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = gen_scores
            self.imp_scores[comp_space + "-" + segmentation_scheme + "-" + comparision_scheme] = imp_scores

            # In this section you will extract both statistical and frequency features (at least 10 from each and concatenate them)
            # from gen_train, gen_test, and imp_test ==> Remember? all these are 2D matrices containing a list of windows
            # For each window, you will extract 10 features, that means you will get 3 new 2D matrices, say
            # gen_train_fmatrix, gen_test_fmatrix, and imp_test_fmatrix; the size of these matrices would be (number of windows, number of features)
            # The gen_train_fmatrix shall serve a the template for the user. Next, you will call get_scores(query, comparision_scheme) function to get the score for query.
            # The query is basically the test vector (rows) of gen_test_fmatrix and imp_test_fmatrix
            # if the query is from gen_test_fmatrix then the score will be appended to gen_scores,
            # else if the query is from imp_test_fmatrix then the score will be appended to imp_scores,
            # Finally, self.gen_scores and self.imp_scores shall be updated with the gen_scores and imp_scores respectively

        else:
            raise ValueError('Comparision space unknown!')
        pass


#######################################################################################################################
# ******************The implementations of ResultAnalysis is for Lab3 ***************************
class ResultAnalysis:
    def __init__(self, Population):
        self.Population = Population

    # A method to normalize gen and imp scores based on the comparison scheme
    def normalize_gen_imp_scores(self, og_gen_scores, og_imp_scores, comparison_scheme):
        if comparison_scheme is not "hist":
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
            processed_gen_scores=np.sort(processed_gen_scores)
            processed_imp_scores=np.sort(processed_imp_scores)
            return (processed_gen_scores, processed_imp_scores)
        else:  # for hist, we don't need to do 1- the values for imp and gen
            gen_score_scaler = MyScaler(og_gen_scores, method="minmax")
            gen_scores = gen_score_scaler.transform(og_gen_scores)
            imp_scores = gen_score_scaler.transform(og_imp_scores)
            return (gen_scores, imp_scores)

    def analyze(self):
        for person in self.Population:
            for key in person.gen_scores:
                comparison_scheme = key.split('-')[2]
                processed_scores = self.normalize_gen_imp_scores(person.gen_scores[key], person.imp_scores[key], comparison_scheme)
                print("KEY",key)
                evaluator = Evaluator(processed_scores[0], processed_scores[1])
                person.heat_map_scores[key + "-EER"] = evaluator.get_eer()
                person.heat_map_scores[key + "-LFRR"] = evaluator.get_frrs()[34]
                person.heat_map_scores[key + "-HFAR"] = evaluator.get_fars()[94]
                evaluator.plot_roc()
                evaluator.plot_det()
                evaluator.get_eer()
                evaluator.get_hists_overlap()
        # The Population consists of objects, each object represents one individual
        # Now use the Evaluator class to analyse the performance of each of the user,
        # and then the performance of the overall biometric system
        # Basically, in lab1 you had only one user and three/four metrics
        # In this lab, we have the following:
        # Users: FIVE
        # Methods: SIX, got why? how many combinations of segmentation and comparison schemes?
        # Metric: roc, det, eer, and hist_intersection
        # What is the best way to compare the SIX methods, with three different metrics, on five (or n) users?
        # The aim is to compare the best configuration:
        # I want you to decide on this part, this is an open question, you can think of tables, bar graphs, heatmaps,
        # or any type of graphs you want to use, just achieve the goal?

        # Code for creating the HeatMap
        eer_heat_map = pd.DataFrame()
        lfar_heat_map = pd.DataFrame()
        hfar_heat_map = pd.DataFrame()
        for person in self.Population:
            eer_df = pd.DataFrame([v*100 for k, v in person.heat_map_scores.items() if (k[-3:] == "EER")], columns=[person.user_name], index=[k[:-4] for k, v in person.heat_map_scores.items() if (k[-3:] == "EER")])
            eer_heat_map = pd.concat([eer_heat_map, eer_df], axis=1)
            lfar_df = pd.DataFrame([v*100 for k, v in person.heat_map_scores.items() if (k[-4:] == "LFRR")], columns=[person.user_name], index=[k[:-5] for k, v in person.heat_map_scores.items() if (k[-4:] == "LFRR")])
            lfar_heat_map = pd.concat([lfar_heat_map, lfar_df], axis=1)
            hfar_df = pd.DataFrame([v*100 for k, v in person.heat_map_scores.items() if (k[-4:] == "HFAR")], columns=[person.user_name], index=[k[:-5] for k, v in person.heat_map_scores.items() if (k[-4:] == "HFAR")])
            hfar_heat_map = pd.concat([hfar_heat_map, hfar_df], axis=1)
        print(eer_heat_map)
        print(lfar_heat_map)
        print(hfar_heat_map)
        sns.set(font_scale=0.9)
        fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(10, 6))
        cm = sns.cubehelix_palette(6)
        p1 = sns.heatmap(eer_heat_map, annot=True, cmap=cm, ax=ax[0], annot_kws={"size": 8}, vmin=0, vmax=100,
                         fmt='.0f', linewidths=0.1, cbar=True, cbar_kws=dict(use_gridspec=False, location="top"))
        plt.xlabel("Users")
        plt.ylabel("Classifiers")
        ax[0].set_title('EER Values for different Users')
        p2 = sns.heatmap(lfar_heat_map, annot=True, cmap=cm, ax=ax[1], annot_kws={"size":8}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=False)
        plt.xlabel("Users")
        plt.ylabel("Classifiers")
        ax[1].set_title('FRR Values for threshold set at 0.35')
        p3 = sns.heatmap(hfar_heat_map, annot=True, cmap=cm, ax=ax[2], annot_kws={"size":8}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=False)
        plt.xlabel("Users")
        plt.ylabel("Classifiers")
        ax[2].set_title('FAR Values for threshold set at 0.95')
        plt.subplots_adjust(left=0.29, bottom=0.17, right=.98, top=0.80, wspace=0.20, hspace=0.20)

        plt.show()
        pass


#######################################################################################################################
if __name__ == "__main__":
    data_folder_name = 'SmartwatchData'
    data_path = os.path.join(Path(__file__).parents[1], 'Storage', data_folder_name)
    user_list = os.listdir(data_path)
    # user_list.remove('Data_description.txt')  # Removing Data_description.txt

    # Preparing the data
    # Creating blank dataframes
    raw_data_training = pd.DataFrame()
    raw_data_testing = pd.DataFrame()
    for user in user_list:
        print(f'working on {user}')
        # Training dataframe
        # Reading the Accelerometer_Data from training folder
        Training = pd.read_csv(os.path.join(data_path, user, 'Training', 'Accelerometer_Data.txt'), sep=',',
                               header=None, names=Params.column_names)
        # Using just AccX
        raw_data_training[user] = Training['x']

        # Testing dataframe
        # Reading the Accelerometer_Data from testing folder
        Testing = pd.read_csv(os.path.join(data_path, user, 'Testing', 'Accelerometer_Data.txt'), sep=',', header=None,
                              names=Params.column_names)
        # Using just AccX
        raw_data_testing[user] = Testing['x']

    # Uncomment the following to see what is in the raw_data_training and raw_data_testing, some rows will have NaNs
    # print(raw_data_training.to_string())
    # print(raw_data_testing.to_string())

    # The number of samples in each user is not the same which would bring NaN while concatinating the dataframes
    # The best hack at the moment is to remove those rows with NaNs
    raw_data_training = raw_data_training.dropna()
    raw_data_testing = raw_data_testing.dropna()

    # Do you wanna check if the rows with NaNs are gone are not, uncomment the following
    # print(raw_data_training.to_string())
    # print(raw_data_testing.to_string())

    # Creating a list of objects, considering each user as an object
    People = []
    # user_list = ['User1']  # For the sake of pilot testing, just comment this to test for all users
    for user in user_list:
        gen_train = raw_data_training[user]
        gen_test = raw_data_testing[user]
        imp_combined = pd.concat([raw_data_training, raw_data_testing], ignore_index=True,
                                 sort=False)  # Concatinating train and test both

        # Changing all the users data to one column impostor, just a quick fix.. may have much better solutions
        imp_melted = pd.melt(imp_combined, id_vars=user, value_vars=list(imp_combined.columns).remove(user),
                             var_name="id", value_name="Impostor")
        # print(imp_melted.to_string())

        imp_test = imp_melted.drop([user, 'id'], axis=1)  # Dropping the current user and the id columns and
        # considering all others as impostor,
        # print(imp_test.to_string())
        # To see the details, uncomment the following
        # print(gen_train.to_string())
        # print(gen_test.to_string())
        # print(imp_test.to_string())

        current_user = Person(user, gen_train, gen_test, imp_test)
        for comp_space in Params.comparison_space:
            # We will do the "cycle" and "sliding_window" scheme for only raw_data level comparision
            Params.segmentation_schemes = ['sliding_window', 'cycle'] if comp_space == 'raw_data' else ['sliding_window']
            for seg_scheme in Params.segmentation_schemes:
                for comp_scheme in Params.comparison_schemes:
                    # The following function is responsible for getting the gen and imp scores set for each combination of
                    # segmentation and comparator schemes
                    current_user.set_math_scores(comp_space, seg_scheme, comp_scheme)

        # Saving the individual performance details in a list of objects that we can traverse later
        People.append(current_user)

    # Once we have evaluated each user and have set the gen and imp scores for all cases, now the time to analyse the
    # cases and see which scheme is the best performer. For more details see analyze() in Analyzer class
    Analyzer = ResultAnalysis(People)
    Analyzer.analyze()
#######################################################################################################################