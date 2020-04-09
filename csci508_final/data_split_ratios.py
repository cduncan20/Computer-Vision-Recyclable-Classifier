def interface():
    print("Data Split Manager")
    print("")
    print("Choose how you would like to split your Training, Validation, & Test Data. Default values are 60% Training, "
          "20% Validation, and 20% Testing. However, ensure ratio values are between 0.0 and 1.0 and sum to a value of "
          "1.0")
    print("")
    train_ratio, valid_ratio, test_ratio = split_data()
    return train_ratio, valid_ratio, test_ratio


def initialize_with_default_values():
    print("Initializing with default data split ratios. Training = 60%, Validation = 20%, and Test = 20% of available "
          "data")
    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2

    return train_ratio, valid_ratio, test_ratio


def select_new_values():
    while True:
        train_ratio = float(input('Training Data Ratio: '))
        valid_ratio = float(input('Validation Data Ratio: '))

        if (train_ratio + valid_ratio) > 0.99:
            print("The selected values are too large as the test set is too small. Please select new smaller values")
            continue
        else:
            test_ratio = 1.0 - (train_ratio + valid_ratio)

            print("You have selected to split data as follows:")
            print("{}% Training, {}% Validation, {}% Testing.".format(train_ratio * 100,
                                                                      valid_ratio * 100,
                                                                      test_ratio * 100))
            confirmation = input("Is this correct? (y/n): ")
            while confirmation not in 'yn':
                confirmation = input("Invalid argument provided. Please select a valid option. Are the above ratios "
                                     "correct?")

            if confirmation == 'y':
                break
            else:
                print("Please select new values")
                continue

    return train_ratio, valid_ratio, test_ratio


def split_data():
    selected_method = input("Would you like to change the data split ratios? (y/n): ")
    while selected_method not in 'yn':
        selected_method = input("Invalid argument provided. Please select a valid option. Would you like to change the "
                                "ratios? y/n")

    if selected_method == 'y':
        train_ratio, valid_ratio, test_ratio = select_new_values()
        return train_ratio, valid_ratio, test_ratio
    else:
        print("Exiting Data Split Manager with default ratios")
        train_ratio, valid_ratio, test_ratio = initialize_with_default_values()
        return train_ratio, valid_ratio, test_ratio
