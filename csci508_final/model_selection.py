def interface():
    print("Model Selection Manager")
    print("")
    print("Select the image transformations you would like to add to the data. You may only select one.")
    print("")
    print("Options:")
    print("[1] Resent18 -- Pre-trained")
    print("[2] Resent34 -- Pre-trained")
    print("[3] Resent50 -- Pre-trained")
    print("")

    model_dict = initialize_model_selection_dictionary()

    selected_model = input('Select model: ')
    print("")
    print("Selected model: ")
    if selected_model == '1':
        model_dict['resnet18'] = True
        print("[1] Resent18 -- Pre-trained")
    elif selected_model == '2':
        model_dict['resnet34'] = True
        print("[2] Resent34 -- Pre-trained")
    elif selected_model == '3':
        model_dict['resnet50'] = True
        print("[3] Resent50 -- Pre-trained")
    else:
        print("Invalid Augmentation key: {}".format(selected_model))

    print("")
    return model_dict


def initialize_model_selection_dictionary():
    model_dict = dict([('resnet18', False),
                       ('resnet34', False),
                       ('resnet50', False)])

    return model_dict
