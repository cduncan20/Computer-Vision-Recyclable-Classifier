#!/usr/bin/env python3.6

import argparse
import sys

import csci508_final as csci


def main():
    args = get_args()

    if args.train:
        if args.split_data:
            train_ratio, valid_ratio, test_ratio = csci.data_split_ratios.interface()
        else:
            train_ratio, valid_ratio, test_ratio = csci.data_split_ratios.initialize_with_default_values()

        if args.augment_data:
            augmentations = csci.data_augmentation.interface()
        else:
            augmentations = csci.data_augmentation.initialize_augmentation_dictionary()
            print("")
            print("No transforms will be applied to training data by default.")
            print("")

        if args.epoch_qty:
            epoch_qty = csci.epoch_quantity.interface()
        else:
            epoch_qty = csci.epoch_quantity.initialize_with_default_values()
            
        model_dict, model_file_name = csci.model_selection.interface()
        csci.train_test_model.main(augmentations, model_dict, model_file_name, train_ratio, valid_ratio, test_ratio, epoch_qty)
    else:
        print("In order to train and validate a model please re-execute this program passing the --train flag after "
              "selecting desired settings. For additonal help please re-execute with the --help flag or refer to the "
              "README")


def get_args():
    parser = argparse.ArgumentParser()

    # The command line arguments now available
    parser.add_argument('-a',
                        '--augment-data',
                        action='store_true',
                        default=False,
                        help='Allows user to choose augmentations (flip horizontally, flip vertically, rotate, etc.)'
                             'performed on training data')
    parser.add_argument('-s',
                        '--split-data',
                        action='store_true',
                        default=False,
                        help='Allows user to choose how to split Training, Validation, & Test data')
    parser.add_argument('-e',
                        '--epoch-qty',
                        action='store_true',
                        default=False,
                        help='Allows user to choose the number of epochs for training the selected model')
    parser.add_argument('--train',
                        action="store_true",
                        default=False,
                        help='Trains the chosen model to classify dataset (defined in load_data.py)')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    sys.exit(main())
