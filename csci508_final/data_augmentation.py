def interface():
    print("Data Augmentation Manager")
    print("")
    print("Select the image transformations you would like to add to the data. You may select as many as you like. "
          "Please be sure to put spaces between your selections.")
    print("")
    print("Options:")
    print("[1] Random horizontal flip")
    print("[2] Random vertical flip")
    print("[3] Random +/-30 degree rotation")
    print("[4] Add artificial noise")
    print("[5] Blur image")
    print("[6] All")
    print("[n] None")
    print("")

    selected_augs = input('Select transforms: ')
    print("")
    print("Selected transforms: ")

    aug_dict = initialize_augmentation_dictionary()

    aug_list = selected_augs.split()
    for augmentation in aug_list:
        if augmentation == '1':
            aug_dict['horizontal'] = True
            print("[1] Random horizontal flip")
            continue
        if augmentation == '2':
            aug_dict['vertical'] = True
            print("[2] Random vertical flip")
            continue
        if augmentation == '3':
            aug_dict['rot30'] = True
            print("[3] Random +/-30 degree rotation")
            continue
        if augmentation == '4':
            aug_dict['noise'] = True
            print("[4] Add artificial noise")
            continue
        if augmentation == '5':
            aug_dict['blur'] = True
            print("[5] Blur image")
            continue
        if augmentation == '6':
            aug_dict = {value: True for value in aug_dict}
            print("[6] All")
            continue
        if augmentation == 'n':
            aug_dict = {value: False for value in aug_dict}
            print("[n] No transforms selected")
        else:
            print("Invalid Augmentation key: {}".format(augmentation))

    print("")
    return aug_dict


def initialize_augmentation_dictionary():
    aug_dict = dict([('horizontal', False),
                     ('vertical', False),
                     ('rot30', False),
                     ('noise', False),
                     ('blur', False)])

    return aug_dict
