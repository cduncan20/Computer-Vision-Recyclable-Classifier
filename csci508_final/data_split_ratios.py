import sys


def interface():
    print("Data Split Manager")
    print("")
    print("Choose how you would like to split your Training, Validation, & Test Data."
            "Default values are 60% Training, 20% Validation, and 20% Testing."
            "You can also enter custom values. "
            "However, ensure ratio values are between 0.0 and 1.0 and sum to a value of 1.0")
    print("")
    print("Options:")
    print("Default [d], Custom [c]")
    print("")

    selected_method = input('Select Method: ')

    if selected_method == 'd':
        train_ratio = 0.6
        valid_ratio = 0.2
        test_ratio = 0.2
    elif selected_method == 'c':
        train_ratio = float(input('Training Data Ratio: '))
        valid_ratio = float(input('Validation Data Ratio: '))
        test_ratio = float(input('Testing Data Ratio: '))
    else:
        sys.exit("Invalid Augmentation key: {}".format(selected_method))

    print("")
    print("User chose to split data as follows:")
    print("{}% Training, {}% Validation, {}% Testing".format(train_ratio*100, valid_ratio*100, test_ratio*100))
    print("")

    return train_ratio, valid_ratio, test_ratio
