def interface():
    print("Epoch Quantity Selection Manager")
    print("")
    print("Choose how many epochs you would like to trains your model on. Default value is 5 epochs.")
    print("")
    epoch_qty = choose_epochs_qty()
    return epoch_qty


def initialize_with_default_values():
    print("Initializing with the default epoch quantity of 5 epochs.")
    epoch_qty = 5

    return epoch_qty


def select_new_values():
    while True:
        epoch_qty = int(input('Epoch Quantity: '))

        print("You have selected to train using {} epochs".format(epoch_qty))
        confirmation = input("Is this correct? (y/n): ")
        while confirmation not in 'yn':
            confirmation = input("Invalid argument provided. Please select a valid option. Is the number "
                                    "of epochs entered above correct?")

        if confirmation == 'y':
            break
        else:
            print("Please select new values")
            continue

    return epoch_qty


def choose_epochs_qty():
    selected_method = input("Would you like to change the epoch quantity from the default value? (y/n): ")
    while selected_method not in 'yn':
        selected_method = input("Invalid argument provided. Please select a valid option. Would you like to change the "
                                "epoch quantity from the default value? y/n")

    if selected_method == 'y':
        epoch_qty = select_new_values()
        return epoch_qty
    else:
        print("Exiting Epoch Quantity Selection Manager with default epoch quantity")
        epoch_qty = initialize_with_default_values()
        return epoch_qty
