def interface():
    print("Data Augmentation Manager")
    print("")
    print("Select the image transformations you would like to add to the data. You may select as many as you like. "
          "Please be sure to put spaces between your selections.")
    print("")
    print("Options:")
    print("Flip horizontally [1], Flip vertically [2], Flip vertically and horizontally [3], Rotate 45 degrees [4], "
          "Add artificial noise [5], Blur image [6], All [7], Exit [n] ")
    print("")

    selected_augs = input('Selected transforms: ')

    aug_dict = initialize_augmentation_dictionary()

    aug_list = selected_augs.split()
    for augmentation in aug_list:
        if augmentation == '1':
            aug_dict['horizontal'] = True
            continue
        if augmentation == '2':
            aug_dict['vertical'] = True
            continue
        if augmentation == '3':
            aug_dict['horizontal-vertical'] = True
            continue
        if augmentation == '4':
            aug_dict['rot45'] = True
            continue
        if augmentation == '5':
            aug_dict['noise'] = True
            continue
        if augmentation == '6':
            aug_dict['blur'] = True
            continue
        if augmentation == '7':
            aug_dict = {value: True for value in aug_dict}
            continue
        if augmentation == 'n':
            aug_dict = {value: False for value in aug_dict}
        else:
            print("Invalid Augmentation key: {}".format(augmentation))

    return aug_dict


def initialize_augmentation_dictionary():
    aug_dict = dict([('horizontal', False),
                     ('vertical', False),
                     ('horizontal-vertical', False),
                     ('rot45', False),
                     ('noise', False),
                     ('blur', False)])

    return aug_dict
