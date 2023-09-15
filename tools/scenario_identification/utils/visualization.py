import itertools
from tqdm import tqdm
from matplotlib import pyplot as plt
import uuid

from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

import numpy as np

COLORS = {'lane': 'k', 'crosswalk': 'cyan', 'speed_bump': 'orange', 'road_edge': 'k', 'road_line': 'k'}
ALPHAS = {'lane': 0.1, 'crosswalk': 0.6, 'speed_bump': 0.6, 'road_edge': 0.1, 'road_line': 0.1}
AGENT_COLOR = {'TYPE_VEHICLE': "blue", 'TYPE_PEDESTRIAN': "green", 'TYPE_CYCLIST': "cyan"}

def get_color_map(num_colors):
    import matplotlib.colors as mcolors
    color_dict = mcolors.CSS4_COLORS
    max_colors = len(color_dict.keys())
    assert num_colors > 0 and num_colors <= len(color_dict.keys()), \
        f"Max. num, of colors is {max_colors}; requested {num_colors}"
    
    color_map = {}
    for i, (k, v) in enumerate(color_dict.items()):
        if i > num_colors:
            break
        color_map[i] = k
    return color_map

def plot_cluster_overlap(num_clusters, num_components, labels, scores, shards, tag):
    fig, ax = plt.subplots(num_clusters, num_clusters, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for a in ax.reshape(-1):
        a.set_xticks([])
        a.set_yticks([])

    for i, j in itertools.combinations(range(num_clusters), 2):
        color_i, color_j = 'blue', 'orange'
        idx_i = np.where(labels == i)
        ax[i, j].scatter(scores[idx_i, 0], scores[idx_i, 1], color=color_i)
        idx_j = np.where(labels == j)
        ax[i, j].scatter(scores[idx_j, 0], scores[idx_j, 1], color=color_j)

    filename = f"{tag}_kmeans-{num_clusters}_pca-{num_components}_overlap_shards{shards[0]}-{shards[-1]}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_polylines(polylines, road_graph, ax = None, num_windows = 0, color='k', alpha=1.0, linewidth=0.5, dim=2):
    polyline_pos_list = []
    for pl in polylines:
        start_idx, end_idx = pl['polyline_index']
        polyline_pos = road_graph[start_idx:end_idx, :dim]
        polyline_pos_list.append(polyline_pos)
        if ax is None:
            continue
        if num_windows == 1:
            ax.plot(polyline_pos[:, 0], polyline_pos[:, 1], color, alpha=alpha, linewidth=linewidth, ms=2)
        else:
            for a in ax.reshape(-1):
                a.plot(polyline_pos[:, 0], polyline_pos[:, 1], color, alpha=alpha, linewidth=linewidth, ms=2)
    return polyline_pos_list

def plot_stop_signs(stop_signs, ax = None, num_windows = 0, color='red', dim=2):
    stop_sign_xy = np.zeros(shape=(len(stop_signs), dim))
    for i, stop_sign in enumerate(stop_signs):
        pos = stop_sign['position']
        stop_sign_xy[i] = pos[:dim]
        if ax is None:
            continue
        
        if num_windows == 1:
            ax.scatter(pos[0], pos[1], s=16, c=color, marker='H', alpha=1.0)
        else:
            for a in ax.reshape(-1):
                a.scatter(pos[0], pos[1], s=16, c=color, marker='H', alpha=1.0)

    return stop_sign_xy

def plot_static_map_infos(
    map_infos, ax = None, num_windows = 0, 
    keys = ['lane', 'stop_sign', 'crosswalk', 'speed_bump', 'road_edge', 'road_line'], dim=2
):
    road_graph = map_infos['all_polylines'][:, :dim]
    map_infos_pos = {}
    for key in keys:
        if key not in map_infos.keys():
            continue
        
        if key == 'stop_sign':
            map_infos_pos[key] = plot_stop_signs(map_infos[key], ax, num_windows, dim=dim)
        else:
            map_infos_pos[key] = plot_polylines(
                map_infos[key], road_graph, ax, num_windows, color=COLORS[key], alpha=ALPHAS[key], dim=dim)
                    
    return map_infos_pos


def plot_dynamic_map_infos(map_infos, ax = None, num_windows = 0, keys = ['stop_point'], dim=2):
    map_infos_pos = {}
    for key in keys:
        if key not in map_infos.keys():
            continue
        
        if key == 'stop_point':
            if len(map_infos[key]) <= 0:
                continue
        
            stop_points = map_infos[key][0]
            for i in range(stop_points.shape[1]):
                pos = stop_points[0, i, :2]
                if ax is None:
                    continue

                if num_windows == 1:
                    ax.scatter(pos[0], pos[1], s=6, c='purple', marker='s', alpha=1.0)
                else:
                    for a in ax.reshape(-1):
                        a.scatter(pos[0], pos[1], s=6, c='purple', marker='s', alpha=1.0)

        # if key == 'stop_sign':
        #     map_infos_pos[key] = plot_stop_signs(map_infos[key], ax)
        # else:
        #     map_infos_pos[key] = plot_polylines(
        #         map_infos[key], road_graph, ax, color=COLORS[key], alpha=ALPHAS[key])
                    
    return map_infos_pos

def plot_lanes_by_distance(lanes, order, dists, ax, k=-1):
    if k == -1:
        # ndists = 1 - np.clip((dists - dists.mean()) / dists.std(), 0.0, 0.1)
        ndists = 1 - np.clip(dists / dists.max(), 0.0, 1.0)
        for lane_idx in order:
            lane = lanes[lane_idx].T
            ax.plot(
                lane[0], lane[1], c=cm.winter(ndists[lane_idx]), alpha=ndists[lane_idx], linewidth=0.5)
    else:
        order = order[:k]
        dists = dists[order]
        # ndists = 1 - np.clip((dists - dists.mean()) / dists.std(), 0.0, 0.1)
        ndists = 1 - np.clip(dists / dists.max(), 0.0, 1.0)

        for i, lane_idx in enumerate(order):
            lane = lanes[lane_idx].T
            ax.plot(
                lane[0], lane[1], c=cm.winter(ndists[i]), alpha=1.0, linewidth=0.5)

def plot_interaction(ax, ax_idx, pos_i, agent_type_i, i_idx, traj_i, pos_j, agent_type_j, j_idx, traj_j, title):
    ax[ax_idx].scatter(
        pos_i[0, 0], pos_i[0, 1], color=AGENT_COLOR[agent_type_i], marker='*', s=10, label='Start')
    ax[ax_idx].plot(pos_i[:, 0], pos_i[:, 1], color=AGENT_COLOR[agent_type_i], alpha=0.6, linewidth=1)

    ax[ax_idx].scatter(pos_j[0, 0], pos_j[0, 1], color=AGENT_COLOR[agent_type_j], marker='*', s=10)
    ax[ax_idx].plot(
        pos_j[:, 0], pos_j[:, 1], color=AGENT_COLOR[agent_type_j], alpha=1, linewidth=1, linestyle='dashed')

    if i_idx != -1:
        ax[ax_idx].scatter(
            traj_i[i_idx, 0], traj_i[i_idx, 1], color='red', marker='+', s=10, label='Conflict Point')
    if j_idx != -1:
        ax[ax_idx].scatter(traj_j[j_idx, 0], traj_j[j_idx, 1], color='red', marker='+', s=10)
    
    ax[ax_idx].legend()
    ax[ax_idx].set_title(title)

# --------------------------------------------------------------------------------------------------
# Unused so far
# --------------------------------------------------------------------------------------------------

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
    ax.axis([-size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y, size / 2 + center_y])
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