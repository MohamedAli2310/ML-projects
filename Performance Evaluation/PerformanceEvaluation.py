# Importing the required packages
# Stick to using only numpy, pandas, and matplotlib packages unless stated otherwise
import numpy as np
from matplotlib import pyplot as plt

def generate_threshold(gen, imp):
    # generate_threshold uses the genuine scores and imposter scores to generate an array of thresholds which would
    # be useful in determining the different FAR, FRR, and TAR for the scores. The method generates the score by
    # using the minimum and maximum scores from both genuine and imposter scores as the range of thresholds. Following
    # which, we create 198 other thresholds (total 200 because of min and max scores) in equal intervals between the
    # minimum and maximum threshold by adding score_increment in an iterative manner.
    min_score = min(imp.min(), gen.min())
    max_score = max(imp.max(), gen.max())
    score_increment = (max_score-min_score)/199
    thresholds = np.ndarray((200,))
    thresholds[0] = min_score
    for i in range(1,200):
        thresholds[i]=thresholds[i-1] + score_increment
    return thresholds

def generate_performance_values(gen_scores, imp_scores):
    # generate_performance_values uses the threshold values (from generate_threshold) and the genuine and imposter
    # scores to generate the TAR, FRR, and FAR values for these different scores. Basically, what it does is to
    # iterate these scores through different threshold values before incrementing the FAR, FRR, or TAR values.
    thresholds = generate_threshold(gen_scores, imp_scores)
    # The following arrays contain the FAR, FRR, and TAR values for the genuine-imposter score pairs for an
    # individual biometric system
    FARarray = np.ndarray((thresholds.size,))
    FRRarray = np.ndarray((thresholds.size,))
    TARarray = np.ndarray((thresholds.size,))

    # iterate through every single threshold score
    for i in range(thresholds.size):
        FAR = 0
        FRR = 0
        # within each single threshold score, compare the genuine and imposter score to the threshold score
        # and add 1 to FRR or TAR or FAR depending on whether it falls below or above the threshold value
        for j in range(gen_scores.size):
            if (gen_scores[j] < thresholds[i]):
                FRR += 1
        for k in range(imp_scores.size):
            if (imp_scores[k] >= thresholds[i]):
                FAR += 1
        # The FAR/TAR/FRR scores have to be divided by the size of the imposter scores before they are added to the
        # arrays which will contain the scores for this individual biometric system
        FAR = FAR / imp_scores.size
        TAR = 1 - (FRR / gen_scores.size)
        FRR = FRR / gen_scores.size
        FARarray[i] = FAR
        FRRarray[i] = FRR
        TARarray[i] = TAR
    return (FARarray, FRRarray, TARarray)

def plot_roc(gen_scores, imp_scores):
    # ROC: Receiver Operating Curve
    # This functions plots an ROC curve for the given match scores
    # Remove/comment the following pass keyword before you start the implementation

    # list of three colors to color three different ROC curves for three different biometric system
    colors = ["green", "blue", "yellow"]
    # iterate through the three different sets of genuine-imposter scores by generating three different
    # performance values and then plotting them out with pyplot
    for i in range(len(gen_scores)):
        FARarray, FRRarray, TARarray = generate_performance_values(gen_scores[i], imp_scores[i])
        plt.plot(FARarray, TARarray, 'go--', linewidth=2, markersize=6, color=colors[i])
    plt.title('ROC Plot')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Accept Rate')
    plt.xlabel('False Accept Rate')
    plt.show()



def plot_det(gen_scores, imp_scores):
    # DET: Decision Error Treadoff
    # This functions plots an DET curve for the given match scores
    # Remove/comment the following pass keyword before you start the implementation

    # list of three colors to color three different ROC curves for three different biometric system
    colors = ["green", "blue", "yellow"]
    # iterate through the three different sets of genuine-imposter scores by generating three different
    # performance values and then plotting them out with pyplot
    for i in range(len(gen_scores)):
        FARarray, FRRarray, TARarray = generate_performance_values(gen_scores[i], imp_scores[i])
        plt.plot(FRRarray, FARarray, 'go--', linewidth=2, markersize=6, color=colors[i])
        # index is the index of the element in the FRRarray where FAR = FRR
        idx = np.argwhere(np.diff(np.sign(FARarray - FRRarray))).flatten()
        idx = idx[1] if idx.size > 1 else idx
        # plot out the point of the intersect (EER) on a separate red graph
        plt.plot(FRRarray[idx], FRRarray[idx], 'ro')
    plt.title('DET Plot')
    plt.plot(FRRarray, FRRarray, 'r--', linewidth=2)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('False Accept Rate')
    plt.xlabel('False Reject Rate')

    plt.show()
def compute_eer(gen_scores, imp_scores):
    # This functions computes the equal error rate for the given match scores
    # Remove/comment the following pass keyword before you start the implementation

    # generate the arrays containing FAR and FRR with the help of generate_performance_values
    # and then finding the index of the elements in both arrays where both values are most similar
    FARarray, FRRarray, TARarray = generate_performance_values(gen_scores, imp_scores)
    idx = np.argwhere(np.diff(np.sign(FARarray - FRRarray))).flatten()
    return FRRarray[idx][0]

def get_hists_overlap(gen_scores, imp_scores):
    # This function returns the intersection of histograms of the input scores
    # Read the following link to understand how can you find the intersection of two histograms
    # (https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html)

    #converts gen_and imp scores into histograms (hist_1 and hist_2) before finding minima, the minimum between the two
    # histograms.
    hist_1, _ = np.histogram(gen_scores, bins=100, range=[0, 1])
    hist_2, _ = np.histogram(imp_scores, bins=100, range=[0, 1])
    minima = np.minimum(hist_1, hist_2)
    # the intersection can be thought of as the percentage of overlap by the imposter onto the genuine scores over the total
    # imposter scores.
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    # plot out the histogram of all imposter scores, genuine scores and the overlap (done with a bar chart)
    plt.hist(imp_scores, bins=100, range=[0, 1])
    plt.hist(gen_scores, bins=100, range=[0, 1])
    plt.bar(np.arange(0, 1, 0.01), minima, 0.01, color="yellow")
    plt.show()
    # intersection is returned as a fraction/decimal to determine how good or bad a biometric system is with respect to
    # the set of probabilistic scores of its imposters
    return intersection


if __name__ == "__main__":
    # Consider the genuine and impostor scores represent the similarity measure
    # That means the higher the score is the better the match
    # 0 <= thresholds <=1
    # 0 <= genuine_scores <= 1, 0 <= impostor_score <= 1

    # A toy example of hitogram, just to play with:
    np.random.seed(1833)  # setting the random seed so the exp are reproducible
    # num_bins = 1000
    # mu_1 = 0.2  # mean of the impostor scores
    # mu_2 = 0.85  # mean of the genuine scores
    # imp = np.random.normal(mu_1, 0.3, 1000)
    # gen = np.random.normal(mu_2, 0.1, 1000)

    # gen = np.asarray([0.5,0.55,0.6,0.65])
    # imp = np.asarray([0.3,0.35,0.4,0.45])

    # plot_roc(gen, imp)
    # plot_det(gen, imp)
    # print(compute_eer(gen, imp))
    # print("imp:", imp)



    # Biometric system B1: Histograms of genuine and imposter scores have no overlap
    # Create the G1, and I1 using the hint given in the toy example

    mu_B1i = 0.1  # mean of the impostor scores
    mu_B1g = 0.95  # mean of the genuine scores
    impB1 = np.random.normal(mu_B1i, 0.1, 1000)
    genB1 = np.random.normal(mu_B1g, 0.1, 1000)


    # Biometric system B2: Histograms of genuine and imposter scores have 5-10% overlap
    # Create the G2, and I2 using the hint given in the toy example

    mu_B2i = 0.3  # mean of the impostor scores
    mu_B2g = 0.8  # mean of the genuine scores
    impB2 = np.random.normal(mu_B2i, 0.2, 1000)
    genB2 = np.random.normal(mu_B2g, 0.1, 1000)


    # Biometric system B3: Histograms of genuine and imposter scores have about 50% overlap
    # Create the G3, and I3 using the hint given in the toy example

    mu_B3i = 0.3  # mean of the impostor scores
    mu_B3g = 0.8  # mean of the genuine scores
    impB3 = np.random.normal(mu_B3i, 0.31, 1000)
    genB3 = np.random.normal(mu_B3g, 0.4, 1000)

    # Compare B1, B2, and B3
    gen_scores = [genB1, genB2, genB3]
    imp_scores = [impB1, impB2, impB3]
    plot_roc(gen_scores, imp_scores)
    plot_det(gen_scores, imp_scores)
    print(get_hists_overlap(genB1, impB1))
    print(get_hists_overlap(genB2, impB2))
    print(get_hists_overlap(genB3, impB3))
    for i in range(len(gen_scores)):
        print(compute_eer(gen_scores[i], imp_scores[i]))