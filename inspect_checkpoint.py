# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script for inspect checkpoint files."""
import argparse
import sys

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app


def get_tensors_from_checkpoint_file(file_name):
    """
    :param file_name: Name of the checkpoint file.
    :return: filters, biases (list, list)

    Prints the tensor names and shapes in the given checkpoint file
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        filters = []
        biases = []
        bad_keys = ['power', 'Adam', 'lr', 'step']
        for key in sorted(var_to_shape_map):
            if not any([e in key for e in bad_keys]):
                tensor = reader.get_tensor(key)
                print(key, tensor.shape)
                if 'kernel' in key:
                    filters.append(tensor.tolist())
                elif 'bias' in key:
                    biases.append(tensor.tolist())
        return filters, biases

    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        suffixes = [".index", ".meta", ".data"]
        if "Data loss" in str(e) and (any([e in file_name for e in suffixes])):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = \
                "It's likely that this is a V2 checkpoint and you need " \
                "to provide the filename *prefix*.  Try removing the '.' " \
                "and extension.  Try: inspect checkpoint --file_name = {}"
            print(v2_file_error_template.format(proposed_file))


def main(unused_argv):
    print("Unused Args: ", unused_argv)
    if not flags.file_name:
        print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
              "[--tensor_name=tensor_to_print]")
        sys.exit(1)
    else:
        get_tensors_from_checkpoint_file(flags.file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name",
        type=str,
        default="",
        help="logdir + {shared prefix between all files in the checkpoint}.")
    flags, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
