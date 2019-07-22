#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import re
import sys
from os import path
from typing import List

import h5py
import numpy
# noinspection PyPackageRequirements
import progressbar

_CSV_FILE_PATTERN = r'\.?\d{8}T\d{6}-\d{8}T\d{6}(?:-\d+)?(?:-\d{8}T\d{6,12})?\.csv'
_DATASET_EXTENSION = '.hdf5'


def _main():
    progressbar.streams.wrap_stderr()
    parser = argparse.ArgumentParser(prog='dataset-creator',
                                     description='Command line utility to create a training dataset in HDF5 format for '
                                                 'the track recurrent neural network from CSV files created by the '
                                                 'classifier.')
    parser.add_argument('input_directory', help='Path of a directory with CSV files to extract tracks.',
                        type=_directory_path)
    parser.add_argument('output',
                        help='Path of the output training dataset file with {} extension.'.format(_DATASET_EXTENSION),
                        type=_output_file_path)
    parser.add_argument('-a', '--append',
                        help='Flag to indicate that an existing HDF5 file can be used and new datasets should be '
                             'appended to it.', action='store_true')
    args = parser.parse_args()
    DatasetCreator(args.input_directory, args.output, args.append).run()


def _directory_path(directory_path: str) -> str:
    """Parse a directory path argument."""
    normalized_directory_path = _normalize_directory_path(directory_path)
    if not path.isdir(normalized_directory_path):
        raise argparse.ArgumentTypeError("'{}' is not a directory".format(directory_path))
    return normalized_directory_path


def _normalize_directory_path(directory_path: str) -> str:
    """Normalize a directory path.

    Resolve user path (~), transform to absolute path and ensure that it ends with a separator."""
    normalized_directory_path = _normalize_file_path(directory_path)
    # Ensure that the path ends with a separator by joining an empty string to it
    return path.join(normalized_directory_path, '')


def _normalize_file_path(file_path: str) -> str:
    """Normalize a file path.

    Resolve user path (~) and transform to absolute path."""
    normalized_file_path = file_path
    if normalized_file_path.startswith('~'):
        normalized_file_path = path.expanduser(normalized_file_path)
    if not path.isabs(normalized_file_path):
        normalized_file_path = path.abspath(normalized_file_path)
    return normalized_file_path


def _output_file_path(output_file_path: str) -> str:
    """Parse an output file path argument."""
    normalized_file_path = _normalize_file_path(output_file_path)
    if path.exists(output_file_path) and not path.isfile(output_file_path):
        raise argparse.ArgumentTypeError("'{}' already exists and is not a file.".format(output_file_path))
    elif path.splitext(output_file_path)[1] != _DATASET_EXTENSION:
        raise argparse.ArgumentTypeError("The output file must have the '{}' extension".format(_DATASET_EXTENSION))
    return normalized_file_path


class DatasetCreator:

    def __init__(self, input_directory_path: str, output_file_path: str, append: bool) -> None:
        super().__init__()
        self._input_directory_path = input_directory_path
        self._output_file_path = output_file_path
        self._append = append
        self._dataset_name = None

    def run(self) -> None:
        """Create the dataset."""
        csv_file_paths = self._find_csv_file_paths()
        self._create_dataset(csv_file_paths)

    def _find_csv_file_paths(self) -> List[str]:
        """Return a list of CSV files used as input to create the dataset."""
        print('Buscando archivos CSV para crear el dataset...')
        csv_files = []
        for root, _, files in os.walk(self._input_directory_path):
            for file in files:
                if re.fullmatch(_CSV_FILE_PATTERN, file):
                    file_path = path.join(root, file)
                    csv_files.append(file_path)
                    print(' - {}'.format(path.relpath(file_path, self._input_directory_path)))
        print('{} archivos encontrados'.format(len(csv_files)))
        if not csv_files:
            sys.exit(1)
        return sorted(csv_files)

    def _create_dataset(self, csv_file_paths: List[str]) -> None:
        """Create the dataset in HDF5 format."""
        if path.exists(self._output_file_path) and not self._append:
            logging.error(
                "'{}' already exists. If you want to append new datasets "
                "to this file use the -a or --append option.".format(self._output_file_path))
            sys.exit(1)
        with h5py.File(self._output_file_path, "a") as output_file:
            output_file.attrs['col0'] = 'left'
            output_file.attrs['col1'] = 'top'
            output_file.attrs['col2'] = 'width'
            output_file.attrs['col3'] = 'height'
            if len(output_file) == 0:
                self._dataset_name = 0
            else:
                self._dataset_name = max(map(int, output_file.keys())) + 1
            for csv_file_path in progressbar.progressbar(csv_file_paths):
                self._process_csv(csv_file_path, output_file)

    def _process_csv(self, csv_file_path: str, output_file: h5py.File) -> None:
        """Read a CSV file, group the bounding boxes by track id and write them in the output file."""
        track_id_to_class = {}
        track_id_to_bounding_boxes = {}
        with open(csv_file_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                track_id = row['Track']
                if track_id not in track_id_to_class:
                    track_id_to_class[track_id] = row['Class']
                    track_id_to_bounding_boxes[track_id] = []
                track_id_to_bounding_boxes[track_id].append(
                    (max(0, int(row['Left'])), max(0, int(row['Top'])), int(row['Width']), int(row['Height'])))
        for track_id, class_name in track_id_to_class.items():
            bounding_boxes = track_id_to_bounding_boxes[track_id]
            dataset_name = str(self._dataset_name)
            self._dataset_name += 1
            data = numpy.array(bounding_boxes, dtype='uint16')
            dataset = output_file.create_dataset(dataset_name,
                                                 dtype='uint16',
                                                 data=data)
            dataset.attrs['class'] = class_name


if __name__ == '__main__':
    _main()
