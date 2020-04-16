import os
import pathlib

def interface():
    print("Model Selection Manager")
    print("")
    print("Select the image transformations you would like to add to the data. You may only select one.")
    print("")
    print("Options:")
    print("[1] Resent18 -- Pre-trained")
    print("[2] Resent34 -- Pre-trained")
    print("[3] Resent50 -- Pre-trained")
    print("[4] Resent50-modnet -- Pre-trained")
    print("")

    model_dict = initialize_model_selection_dictionary()

    selected_model = input('Select model: ')
    while selected_model not in '1234':
        print("Invalid argument provided. Please select a valid option below.")
        print("[1] Resent18 -- Pre-trained")
        print("[2] Resent34 -- Pre-trained")
        print("[3] Resent50 -- Pre-trained")
        print("[4] Resent50-modnet -- Pre-trained")
        print("")
        selected_model = input('Select model: ')

    print("")
    print("Selected model: ")
    if selected_model == '1':
        model_name = 'resnet18'
        model_dict[model_name] = True
        print("[1] Resent18 -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    elif selected_model == '2':
        model_name = 'resnet34'
        model_dict[model_name] = True
        print("[2] Resent34 -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    elif selected_model == '3':
        model_name = 'resnet50'
        model_dict[model_name] = True
        print("[3] Resent50 -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    elif selected_model == '4':
        model_name = 'resnet50-modnet'
        model_dict[model_name] = True
        print("[4] Resent50-modnet -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    else:
        print("Invalid Augmentation key: {}".format(selected_model))

    print("")
    print(model_file_name)
    return model_dict, model_file_name


def initialize_model_selection_dictionary():
    model_dict = dict([('resnet18', False),
                       ('resnet34', False),
                       ('resnet50', False),
                       ('resnet50-modnet', False)])

    return model_dict

def model_file_naming(model_name):
    cwd = pathlib.Path.cwd()
    model_save_path = cwd.joinpath("csci508_final", "saved_models_and_results", "saved_models")

    max_file_num = 0
    for file in os.listdir(model_save_path):
        if file.endswith(".pth"):
            file_strings = file.split("_")
            if model_name == file_strings[0]:
                file_num_string = file_strings[2]
                file_num_string = file_num_string.split(".")
                file_num_string = file_num_string[0]
                file_num = int(file_num_string[1:])
                if file_num > max_file_num:
                    max_file_num = int(file_num)
            
    model_file_name = model_name + "_model_v" + str(max_file_num+1)

    return model_file_name

