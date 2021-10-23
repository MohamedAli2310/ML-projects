## Description

Performance Evaluation practice project to learn more about ROC and DET curves, EER, FAR, FRR, and TAR.
compared the performances of B1, B2, and B3 using the following metrics:
1. ROC plots (https://people.inf.elte.hu/kiss/11dwhdm/roc.pdf)
1. DET plots (similar to ROC except the fact that it plots FAR vs. FRR instead TAR vs. FPR
1. Equal Error Rate (EER): EER is a point on the DET curve where FAR and FRR become equal
1. Overlap of the histograms formed from the genuine and impostor scores of the respective systems

Implemented the plot_roc, plot_det, compute_eer, and get_hists_overlap functions given in PerformanceEvaluation.py 
