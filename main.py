#!/usr/bin/env python3.6

import argparse
import sys

import csci508_final as csci


def main():
    args = get_args()

    if args.split_data:
        train_ratio, valid_ratio, test_ratio = csci.data_split_ratios.interface()

    if args.augment_data:
        augmentations = csci.data_augmentation.interface()
    else:
        augmentations = csci.data_augmentation.initialize_augmentation_dictionary()

    if args.label_data:
        csci.label_data.label_data(augmentations, train_ratio, valid_ratio, test_ratio)

    if args.dry_run:
        csci.pickle_file_test.run_test()

    if args.train:
        print("This is where training will go")

    if args.test:
        print("This is where testing will go")

    # TODO: Tyler -> Figure out a good way to manage having multiple network architectures and having them all use
    #  'train' and 'test' flags


def get_args():
    parser = argparse.ArgumentParser()

    # The command line arguments now available
    parser.add_argument('-s',
                        '--split-data',
                        action='store_true',
                        default=False,
                        help='Allows user to choose how to split Training, Validation, & Test data')
    parser.add_argument('-a',
                        '--augment-data',
                        action='store_true',
                        default=False,
                        help='Provides automated data augmentation to data in the TEST set such as horizontal and '
                             'vertical reflection')
    parser.add_argument('-l',
                        '--label-data',
                        action='store_true',
                        default=False,
                        help='Generates a pickle file from data in the Images directory')
    parser.add_argument('--dry-run',
                        action='store_true',
                        default=False,
                        help='Performs a dry run using only the data in the TRIAL directory')
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
