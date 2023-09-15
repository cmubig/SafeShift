import argparse
import pdb
import os
import pickle as pkl
import pandas as pd
import pingouin as pg

from tqdm import tqdm
from matplotlib import pyplot as plt
from natsort import natsorted
from operator import itemgetter
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

import numpy as np
import itertools


base = os.path.expanduser('~/monet_shared/shared/mtr_process')

# Uses the "new" splits, from resplit.py; this way, test is labeled as well
train_base = f'{base}/new_processed_scenarios_training'
val_base = f'{base}/new_processed_scenarios_validation'
test_base = f'{base}/new_processed_scenarios_testing'
train_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
val_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
test_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

# Taken from: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_motion.ipynb
def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """Compute a color map array of shape [num_agents, 4]."""
    colors = cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

    Args:
        all_states: states of agents as an array of shape [num_agents, num_steps,
        2].
        all_states_mask: binary mask of shape [num_agents, num_steps] for
        `all_states`.

    Returns:
        center_y: float. y coordinate for center of data.
        center_x: float. x coordinate for center of data.
        width: float. Width of data.
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width

def visualize_one_step(states,
                       mask,
                       roadgraph,
                       title,
                       center_y,
                       center_x,
                       width,
                       color_map,
                       size_pixels=1000):
    """Generate visualization for a single step."""

    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # Plot agent current position.
    ax.scatter(
        masked_x,
        masked_y,
        marker='o',
        linewidths=3,
        color=colors,
    )

    # Title.
    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)
    plt.close(fig)
    return image

def create_animation(images):
    """ Creates a Matplotlib animation of the given images.

    Args:
        images: A list of numpy arrays representing the images.

    Returns:
        A matplotlib.animation.Animation.

    Usage:
        anim = create_animation(images)
        anim.save('/tmp/animation.avi')
        HTML(anim.to_html5_video())
    """

    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images) // 2, interval=100)
    plt.close(fig)
    return anim

def visualize_all_agents_smooth(
    decoded_example,
    size_pixels=1000,
):
    """Visualizes all agent predicted trajectories in a serie of images.

    Args:
        decoded_example: Dictionary containing scenario information
        size_pixels: The size in pixels of the output image.

    Returns:
        T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
    """
    # [num_agents, num_past_steps, 2] float32.
    # past_states = tf.stack(
    #     [decoded_example['state/past/x'], decoded_example['state/past/y']],
    #     -1).numpy()
    # past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0
    past_states = decoded_example['track_infos']['trajs'][:, :10, :2]
    past_states_mask = decoded_example['track_infos']['trajs'][:, :10, -1] > 0.0

    # [num_agents, 1, 2] float32.
    # current_states = tf.stack(
    #     [decoded_example['state/current/x'], decoded_example['state/current/y']],
    #     -1).numpy()
    # current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0
    current_states = decoded_example['track_infos']['trajs'][:, 10, :2][:, np.newaxis, :]
    current_states_mask = decoded_example['track_infos']['trajs'][:, 10, -1][:, np.newaxis] > 0.0

    # [num_agents, num_future_steps, 2] float32.
    # future_states = tf.stack(
    #     [decoded_example['state/future/x'], decoded_example['state/future/y']],
    #     -1).numpy()
    # future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0
    future_states = decoded_example['track_infos']['trajs'][:, 11:, :2]
    future_states_mask = decoded_example['track_infos']['trajs'][:, 11:, -1] > 0.0

    # [num_points, 3] float32.
    #roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()
    roadgraph_xyz = decoded_example['map_infos']['all_polylines'][:, :3]

    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    color_map = get_colormap(num_agents)

    # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    # [num_agens, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate([past_states_mask, current_states_mask, future_states_mask], 1) > 0.0

    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    images = []

    # Generate images from past time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(past_states, num_past_steps, 1),
            np.split(past_states_mask, num_past_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'past: %d' % (num_past_steps - i), center_y,
                                center_x, width, color_map, size_pixels)
        images.append(im)

    # Generate one image for the current time step.
    s = current_states
    m = current_states_mask

    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
                            center_x, width, color_map, size_pixels)
    images.append(im)

    # Generate images from future time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, 1),
            np.split(future_states_mask, num_future_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'future: %d' % (i + 1), center_y, center_x, width,
                                color_map, size_pixels)
        images.append(im)

    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train_inputs = natsorted([(x, f'{train_base}/{x}') for x in os.listdir(train_base)])
    val_inputs =  natsorted([(x, f'{val_base}/{x}') for x in os.listdir(val_base)])
    test_inputs =  natsorted([(x, f'{test_base}/{x}') for x in os.listdir(test_base)])
    # Load meta pickle things; takes ~30s
    with open(train_meta, 'rb') as f:
        train_metas = natsorted(pkl.load(f), key=itemgetter(*['scenario_id']))
    with open(val_meta, 'rb') as f:
        val_metas = natsorted(pkl.load(f), key=itemgetter(*['scenario_id']))
    with open(test_meta, 'rb') as f:
        test_metas = natsorted(pkl.load(f), key=itemgetter(*['scenario_id']))
    
    for (_, scenario_path), _ in tqdm(zip(train_inputs, train_metas), 'Processing scenarios...', total=len(train_metas)):
        # input_meta is a dict with the following keys:
        # - scenario_id: id
        # - current_time_index: always 10 (i.e. hist is 0-10 inclusive, fut is 11-90)
        # - timestamps_seconds: list of timestamps, length 91
        # - sdc_track_index: index of self-driving car
        # - objects_of_interest: list of objects of interest, is often empty list
        # - tracks_to_predict: dict {track_index: [], difficulty: [], object_type: []}
        # ###############################################################
        # scenario_path is path to the pickled, processed file saved from data_preprocess.py:
        with open(scenario_path, 'rb') as f:
            scenario = pkl.load(f)
        # scenario is a dict with ALL of the above keys, plus the following:
        # - track_infos: dict {object_id: [], object_type: [], trajs: []}
        #    - the indices here correspond to the indices in e.g. tracks_to_predict and sdc_track_index
        #    - i.e. track_infos[trajs][scenario[sdc_track_index]] is a 91x10 array
        #    - The 10 elements are as follows: 
        #       - center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
        # - map_infos: dict with meta data about each map type
        #   - e.g. map_infos['lane'] is list of all lanes, with info like 'speed_limit_mph' and 'polyline_index']
        #   - polyline_index corresponds to index into map_infos['all_polylines'], where first 3 fields of each are xyz pos.
        # - dynamic_map_infos: dict with information about traffic light states, stop points, etc.
        #   - dynamic_map_infos['state'] = list of length 91, containing state for each lane traffic light
        #   - dynamic_map_infos['lane_id'] = same as above, but lane_id
        #   - dynamic_map_infos['stop_point'] = same as above, but consists of num-lanes x 3, for stop points of each

        sdc_traj = scenario['track_infos']['trajs'][scenario['sdc_track_index']]
        to_predict_trajs = scenario['track_infos']['trajs'][scenario['tracks_to_predict']['track_index']]
        
        import pdb; pdb.set_trace()
        ############################
        # Save video of scene
        # Takes ~15sec to process
        images = visualize_all_agents_smooth(scenario)
        # To do every 5th frame:
        #anim = create_animation(images[::5])
        anim = create_animation(images)
        # Takes ~1.5min as well to process
        anim.save('tmp.gif', dpi=100, writer=PillowWriter(fps=10))
        import pdb; pdb.set_trace()
    