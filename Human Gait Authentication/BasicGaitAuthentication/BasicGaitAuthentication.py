# Install packages as per the need
import numpy as np
import pandas as pd
from dtw import *
from scipy.spatial.distance import cityblock
from scipy import signal
from scipy import interpolate
import itertools as it
from matplotlib import pyplot as plt
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
    segmentation_schemes = ['cycle','sliding_window']
    comparison_schemes = ['pointwise', 'dtw', 'hist']
    num_bins = 100


#######################################################################################################################
class Preprocessor:
    def __init__(self, signal):
        # signal is a one dimensional array, a time series, for example acceleration in dimension x
        self.signal = signal

    def remove_noise(self):
        self.signal = self.signal.rolling(window=3, min_periods=1).mean()
        # print(self.signal.rolling(window=3, min_periods=1)) FOR DEBUGGING
        # print("DONE", self.signal, "END") FOR DEBUGGING

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
                plt.plot(final_segment_array[i][0])
            plt.show()
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
                end = start + (window_segment_size)+1
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
                plt.plot(final_segment_array[i][0])
            plt.show()
            return final_segment_array
        else:
            raise ValueError('Unknown segmentation_scheme!')

    # *************************************** Following implementations are for Lab3 ***************************
    # schemes = ['standard', 'minmax'] # def normalize(self, scheme='standard'): It maybe necessary in some case that
    # you need to normalize the values in a particular range because the diance metric are resolution
    # sensitive....but be very careful while applying normalization because it may distort the values and its
    # uniqueness pass




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

        distance_array = np.zeros((1, self.template.size))
        for i in range(self.template.size):
            template = self.template[i]
            # determines whether template or query is bigger
            bigger_array = template[0] if template[0].size >= self.query[0].size else self.query[0]
            smaller_array = template[0] if template[0].size < self.query[0].size else self.query[0]
            # creates an array for the x-axis (0,1,2,....,smaller_array.size)
            x = np.arange(0, smaller_array.size)
            y = smaller_array
            # extracts the relationship between x and the smaller array
            f = interpolate.interp1d(x, y)
            # creates a new interpolated array of size = bigger_array.size from the smaller array.
            ynew = f(np.linspace(0, smaller_array.size-1, num=bigger_array.size, endpoint=True))
            # finds the distance between the two vectors using linalg.norm
            distance_array[0][i] = np.linalg.norm(bigger_array - ynew)
        # return the median of the distances as the final match score
        return np.median(distance_array)

    def dtw_distance(self):

        distance_array = np.zeros((1, self.template.size))
        i = 0
        for template in self.template:
            bigger_array = template[0] if template[0].size >= self.query[0].size else self.query[0]
            smaller_array = template[0] if template[0].size < self.query[0].size else self.query[0]
            x = np.arange(0, smaller_array.size)
            y = smaller_array
            f = interpolate.interp1d(x, y)
            ynew = f(np.linspace(0, smaller_array.size - 1, num=bigger_array.size, endpoint=True))

            # using dtw-python package with keep_internals=True and distance_only=True to make it faster
            d = dtw(ynew, bigger_array, keep_internals=True, distance_only=True)
            distance_array[0][i] = d.distance
            i += 1
        return np.median(distance_array)

    def hist_distance(self):

        distance_array = np.zeros((1, self.template.size))
        i = 0
        for template in self.template:
            bigger_array = template[0] if template[0].size >= self.query[0].size else self.query[0]
            smaller_array = template[0] if template[0].size < self.query[0].size else self.query[0]
            x = np.arange(0, smaller_array.size)
            y = smaller_array
            f = interpolate.interp1d(x, y)
            ynew = f(np.linspace(0, smaller_array.size - 1, num=bigger_array.size, endpoint=True))
            np.linspace(0, smaller_array.size, num=bigger_array.size)
            distance_array[0][i] = self.hist_distance_helper(bigger_array, ynew)
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

    # SAME AS LAB 1
    def get_fars(self):
        fars_array = np.ndarray((self.thresholds.size, ))
        for threshold in self.thresholds.size:
            fars_count = 0
            for score in self.imp_scores.size:
                if score >= threshold:
                    fars_count += 1
            np.append(fars_array, (fars_count / self.imp_scores.size))
            # returns fars
        return fars_array

    # SAME AS LAB 1
    def get_frrs(self):
        frrs_array = np.ndarray((self.thresholds.size,))
        for threshold in self.thresholds.size:
            frrs_count = 0
            for score in self.gen_scores.size:
                if score < threshold:
                    frrs_count += 1
            np.append(frrs_array, (frrs_count / self.gen_scores.size))
            # returns fars
        return frrs_array

    # SAME AS LAB 1
    def plot_roc(self):
        # list of three colors to color three different ROC curves for three different biometric system
        colors = ["green", "blue", "yellow"]
        # iterate through the three different sets of genuine-imposter scores by generating three different
        # performance values and then plotting them out with pyplot
        tar_array = np.ndarray((self.thresholds.size,))
        for score in self.get_frrs():
            np.append(tar_array, 1 - score)
        for i in range(len(self.gen_scores)):
            plt.plot(self.get_fars(), tar_array, 'go--', linewidth=2, markersize=6, color=colors[i])
        plt.title('ROC Plot')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Accept Rate')
        plt.xlabel('False Accept Rate')
        plt.show()

    # SAME AS LAB 1
    def plot_det(self):
        # list of three colors to color three different ROC curves for three different biometric system
        colors = ["green", "blue", "yellow"]
        # iterate through the three different sets of genuine-imposter scores by generating three different
        # performance values and then plotting them out with pyplot
        for i in range(len(self.gen_scores.size)):
            plt.plot(self.get_frrs(), self.get_fars(), 'go--', linewidth=2, markersize=6, color=colors[i])
            # index is the index of the element in the FRRarray where FAR = FRR
            idx = np.argwhere(np.diff(np.sign(self.get_fars() - self.get_frrs()))).flatten()
            idx = idx[1] if idx.size > 1 else idx
            # plot out the point of the intersect (EER) on a separate red graph
            plt.plot(self.get_frrs()[idx], self.get_frrs()[idx], 'ro')
        plt.title('DET Plot')
        plt.plot(self.get_frrs(), self.get_frrs(), 'r--', linewidth=2)
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

    # By default we have segmentation_scheme = 'cycle', comparision_scheme = 'dtw', once you pass the values
    # it the passed values will automatically replace the default values

    def set_math_scores(self, segmentation_scheme='cycle', comparision_scheme='dtw'):
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
        # Creates a Comparator with gen train data
        comparator = Comparators(processed_gen_train)
        gen_scores = np.ndarray((processed_gen_test.size,))
        imp_scores = np.ndarray((processed_imp_test.size,))

        # compares all gen scores with the train data and get scores
        for i in range(processed_gen_test.size):
            gen_scores[i]=comparator.get_scores(processed_gen_test[i], comparision_scheme)
        # compares all imp scores with the train data and get scores
        for i in range(processed_imp_test.size):
            imp_scores[i] = comparator.get_scores(processed_imp_test[i], comparision_scheme)
        # prints imp and gen final median scores
        print ("IMP SCORES",np.median(imp_scores))
        print ("GEN SCORES",np.median(gen_scores))

        self.gen_scores[segmentation_scheme + "-" + comparision_scheme] = gen_scores
        self.imp_scores[segmentation_scheme + "-" + comparision_scheme] = imp_scores

#######################################################################################################################
# ******************The implementations of ResultAnalysis is for Lab3 ***************************
class ResultAnalysis:
    def __init__(self, Population):
        self.Population = Population

    def analyze(self):
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
        for person in self.Population:
            # print("HIT")
            print(person.gen_scores, person.imp_scores)


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
        for seg_scheme in Params.segmentation_schemes:
            for comp_scheme in Params.comparison_schemes:
                # The following function is responsible for getting the gen and imp scores set for each combination of
                # segmentation and comparator schemes
                print(seg_scheme, comp_scheme)
                current_user.set_math_scores(seg_scheme, comp_scheme)
                print(seg_scheme, comp_scheme)

        # Saving the individual performance details in a list of objects that we can traverse later
        People.append(current_user)

    # Once we have evaluated each user and have set the gen and imp scores for all cases, now the time to analyse the
    # cases and see which scheme is the best performer. For more details see analyze() in Analyzer class
    Analyzer = ResultAnalysis(People)
    Analyzer.analyze()

#######################################################################################################################
