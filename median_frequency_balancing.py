import numpy as np

# Take care of the class imbalance
def median_frequency_balancing(labels):
    # Labels are a matrix of not one-hot labels of size (Len of training data x 1)
    Labels = np.unique(labels)
    NumClass = len(Labels)
    ClassWeights = np.zeros((NumClass))
    ClassFreq = np.zeros((1, NumClass))

    # Estimate Class Frequency
    for i in range(NumClass):
        pixel_in_this_class, test = np.where(labels.astype(int) == i)
        ClassFreq[0, i] = len(pixel_in_this_class) / len(labels)

    MedianFreq = np.median(ClassFreq)

    for j in range(NumClass):
        ClassWeights[j] = MedianFreq / ClassFreq[0,j]

    return ClassWeights