#!/bin/bash
bag="${1?"Usage: $0 <input.bag> <output/folder>"}"
output_folder="${2-output}"

echo Extracting $bag

fname=$(basename "$bag")
dataset="${fname%.bag}"
mkdir -p "${output_folder}/${dataset}"
python extract_rosbag_to_tf.py --bag $bag --n_skip 1 --output_folder $output_folder
