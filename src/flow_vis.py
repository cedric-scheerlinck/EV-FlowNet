import argparse
import cv2
import numpy as np
import os
import shutil
import subprocess

from vis_utils import flow_viz_np


def main(args):
    frame_folder = os.path.join(args.output_folder, 'flow_frames')
    if os.path.isdir(frame_folder):
        shutil.rmtree(frame_folder)
    os.makedirs(frame_folder)
    fname_list = [f for f in os.listdir(args.input_folder) if f.endswith('.npy')]
    for fname in fname_list:
        flow = np.load(os.path.join(args.input_folder, fname))
        if flow.shape[0] == 2:
            flow_x = flow[0, ...]
            flow_y = flow[1, ...]
        elif flow.shape[-1] == 2:
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
        else:
            raise Exception('Cannot parse flow shape {}'.format(flow.shape))
        flow_bgr = flow_viz_np(flow_x, flow_y)
        out_name = '.'.join(fname.split('.')[:-1]) + '.png'
        out_path = os.path.join(frame_folder, out_name)
        cv2.imwrite(out_path, flow_bgr)
    ffmpeg = ['ffmpeg', '-y', '-pattern_type', 'glob', '-i',
              os.path.join(frame_folder, '*.png'),
              os.path.join(args.output_folder, 'flow.mp4')]
    subprocess.check_output(ffmpeg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', required=True, type=str)
    parser.add_argument('-o', '--output_folder', required=True, type=str)

    args = parser.parse_args()
    main(args)
