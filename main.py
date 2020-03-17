#!/usr/bin/env python3.6

import argparse
import sys

import csci508_final as csci


def main():
    args = get_args()

    if args.label_data:
        csci.label_data.label_data()

    if args.pickle_test:
        csci.pickle_file_test.run_test()


def get_args():
    parser = argparse.ArgumentParser()

    # The command line arguments now available
    parser.add_argument('-l', '--label_data', action='store_true', default=False, help='Generates a pickle file from '
                                                                                       'data in the Images directory')
    parser.add_argument('-p', '--pickle_test', action='store_true', default=False, help='Runs a test of the pickle file'
                                                                                        ' from the TRIAL set')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    sys.exit(main())
