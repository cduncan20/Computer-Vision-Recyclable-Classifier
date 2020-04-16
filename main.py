#!/usr/bin/env python3.6

import argparse
import sys

import csci508_final as csci


def main():
    args = get_args()

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

    if args.train:
        model_dict = csci.model_selection.interface()
        csci.train_test_model.main(augmentations, model_dict, train_ratio, valid_ratio, test_ratio, epoch_qty)

    if args.test:
        print("This is where testing will go")

    # TODO: Tyler -> Figure out a good way to manage having multiple network architectures and having them all use
    #  'train' and 'test' flags


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
                        help='Trains the neural network to classify images from a generated pickle file')
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help='Tests a trained neural network and reports the results')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    sys.exit(main())
