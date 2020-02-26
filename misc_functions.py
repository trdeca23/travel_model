import numpy as np
import os
from sklearn import metrics
#import scikitplot as skplt
import matplotlib.pyplot as plt


#input: numpy array my_array, and float x
#output: number in array that is in the highest x proportion
#note: this only works roughly if the values in my_array have low granularity
def highest_x_proportion(my_array, x):
    my_array_sorted = np.sort(my_array) #sorts array in increasing order
    return(my_array_sorted[np.amax([0,np.ceil(len(my_array_sorted)*(1-x)).astype(int)-1])])


#input:
#outcome - numpy array or pandas series specifying observed outcomes for a test set
#predicted_prob - 1d numpy array specifying predicted probabilities of the outcome
#top_threshold - the proportion of the sample that you plan to label as highest likelihood based on y_pred
#output_name - the name for the .txt that you will write output to
#Note: this is slightly unwieldy and should be divided into two functions: evaluate_probs and evaluate_levels
def evaluate_model(outcome, predicted_prob, top_threshold, output_path_minus_extension):
    print("\n"*3,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    print("MODEL PERFORMANCE\n",
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    ############# evaluation by predicted probability #############
    #ROC and plot
    fpr, tpr, threshold = metrics.roc_curve(outcome, predicted_prob)
    roc_auc = metrics.auc(fpr, tpr)    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.set_title('Receiver Operating Characteristic')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.savefig(os.path.join(output_path_minus_extension + ".png"))
    print("ROC AUC:", roc_auc, "\n",
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    ############# evaluation by level/threshold #############
    #proportion that you want to predict as positive
    high_lk_flag = predicted_prob > highest_x_proportion(predicted_prob, top_threshold)
    #confusion matrix
    cnfs_mat = metrics.confusion_matrix(outcome, high_lk_flag)
    #skplt.metrics.plot_confusion_matrix(y_test, y_pred)
    #plt.show()
    FP = cnfs_mat.sum(axis=0) - np.diag(cnfs_mat) #number of false positives for each category (low likelihood / high likelihood)
    FN = cnfs_mat.sum(axis=1) - np.diag(cnfs_mat) #numer of false negatives
    TP = np.diag(cnfs_mat) #number of true positives
    TN = cnfs_mat.sum() - (FP + FN + TP) #number of true negatives
    FP = FP.astype(float)[1] #we are interested in the second category (high likelihood)
    FN = FN.astype(float)[1]
    TP = TP.astype(float)[1]
    TN = TN.astype(float)[1]
    print("True positives #:", TP, "\n",
          "False positives #:", FP, "\n",
          "True negatives #:", TN, "\n",
          "False negatives #:", FN, "\n", sep = "",
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("True positive rate:", TPR,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print("True negative rate:", TNR,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("Positive predictive value:", PPV,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # Negative predictive value
    NPV = TN/(TN+FN)
    print("Negative predictive value:", NPV,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("False positive rate:", FPR,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # False negative rate
    FNR = FN/(TP+FN)
    print("False negative rate:", FNR,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # False discovery rate
    FDR = FP/(TP+FP)
    print("False discovery rate:", FDR,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN) #same as metrics.accuracy_score(outcome, high_lk_flag)
    print("Accuracy:", ACC,
          file=open(os.path.join(output_path_minus_extension + ".txt"), "a"))

