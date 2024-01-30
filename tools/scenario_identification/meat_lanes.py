import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time
import pandas as pd
# from collections import deque
# from queue import PriorityQueue
import heapq

from natsort import natsorted
from operator import itemgetter
from tqdm import tqdm
from matplotlib import cm
from intervaltree import IntervalTree, Interval
from skspatial.objects import Vector

from tools.scenario_identification.utils.common import (
    POS_XYZ_IDX, CACHE_DIR, VISUAL_OUT_DIR, POS_XY_IDX, VEL_XY_IDX, HEADING_IDX
)
from tools.scenario_identification.utils.visualization import plot_static_map_infos

from closest_lanes import compute_k_closest_lanes

VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, f"{os.path.basename(__file__).split('.')[0]}")
os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)

all_types = ['TYPE_VEHICLE', 'TYPE_CYCLIST', 'TYPE_PEDESTRIAN', 'TYPE_VEHICLE_STATIONARY', 'TYPE_CYCLIST_STATIONARY', 'TYPE_PEDESTRIAN_STATIONARY']

def build_lane_graph(lanes):
    """ Map lane index to lane ID and build a connected graph using entry and exit lanes. """
    id_to_idx = {}
    for l, lane in enumerate(lanes):
        id_to_idx[lane['id']] = l

    graph = {}
    for l, lane in enumerate(lanes):
        # Should always be able to connect back to self
        connected_lanes = [lane['id']] + lane['entry_lanes'] + lane['exit_lanes']
        # Make unique just in case
        connected_lanes = list(set(connected_lanes))
        graph[id_to_idx[lane['id']]] = [id_to_idx[c] for c in connected_lanes]
    return graph

def build_lane_sequences(closest_lanes_per_timestep, lane_graph, lane_segments,
                         dist_threshold, prob_threshold, angle_thresh,
                         traj, traj_vel, traj_heading, track_infos, agent_n, timeout):
    start_time = time.time()
    T, K, D = closest_lanes_per_timestep.shape
    obj_type = track_infos['object_type'][agent_n]
    #follow_rules = obj_type in ['TYPE_VEHICLE']
    #print(f'\t{agent_n}, threshold: {dist_threshold}')

    lane_probs = closest_lanes_per_timestep[..., 0]
    lane_probs = np.maximum(0, 1 - lane_probs/dist_threshold)
    # Assign list of valid regions
    possible_lanes = []
    # in 2d at least
    def get_angle(current_pos, current_heading, current_vel, lane: int):
        lane_segment = lane_segments[lane]
        segment_dists = np.linalg.norm(current_pos[np.newaxis, np.newaxis, :] - lane_segment, axis=-1).mean(axis=-1)
        # i.e. only one point in lane, unable to determine direction or projection things
        if not len(segment_dists):
            return 0, False
        closest_segment = lane_segment[segment_dists.argmin(axis=-1)]
        segment_dir = closest_segment[1] - closest_segment[0]
        current_heading_vec = Vector(np.array([np.cos(current_heading), np.sin(current_heading)]).squeeze())
        angle_diff = np.rad2deg(current_heading_vec.angle_signed(segment_dir[:2]))
        possible_trans = np.abs(angle_diff) < angle_thresh or np.abs(angle_diff) > (180 - angle_thresh)
        # TODO: assess if this is an okay transition if at slow speeds
        return angle_diff, (possible_trans or np.linalg.norm(current_vel) < 1.0)
        # return angle_diff, possible_trans

    for t in range(T):
        dists = {}
        for lane_info in closest_lanes_per_timestep[t, lane_probs[t] > prob_threshold]:
            angle_diff, possible_trans = get_angle(traj[t], traj_heading[t], traj_vel[t], int(lane_info[-1]))
            #if possible_trans or not follow_rules:
            if possible_trans:
                dists[lane_info[-1]] = lane_info
            # dists[lane_info[-1]] = lane_info
        possible_lanes.append(dists)

    # Build interval tree
    active_intervals = {}
    interval_list = []
    for t in range(T):
        # Check for interval ends first
        to_remove = []
        for active, (start, lane_id, lane_infos) in active_intervals.items():
            if active in possible_lanes[t]:
                lane_infos.append(possible_lanes[t][active])
            if active not in possible_lanes[t]:
                end = t
                # half-open interval, start inclusive, end exclusive
                interval_list.append([start, end, (lane_id, np.stack(lane_infos))])
                to_remove.append(active)
        for x in to_remove:
            active_intervals.pop(x)
        
        for lane_id, lane_info in possible_lanes[t].items():
            if lane_id not in active_intervals:
                active_intervals[lane_id] = (t, lane_id, [lane_info])
    for active, (start, lane_id, lane_infos) in active_intervals.items():
        end = T
        interval_list.append([start, end, (lane_id, np.stack(lane_infos))])
    tree = IntervalTree()
    for interval in interval_list:
        start, end = interval[0], interval[1]
        #sub_intervals = np.array_split(np.arange(start, end), factor)
        sub_intervals = np.array_split(np.arange(start, end), 1)
        for sub_interval in sub_intervals:
            if not len(sub_interval):
                break
            sub_start, sub_end = sub_interval[0], sub_interval[-1] + 1
            sub_data = (interval[2][0], interval[2][1][sub_start-start:sub_end-start])
            tree[sub_start:sub_end] = sub_data
    
    # Perform DFS
    # greedy i.e. so that the first one found is heuristically the "best" one
    # PQ = PriorityQueue()
    PQ = []
    # Priority = dist so far + dist if you were to add current interval
    def enqueue_item(interval, cur_seq):
        tot_dist = 0
        tot_n = 0
        if len(cur_seq):
            tot_n += sum([x.end - x.begin for x in cur_seq])
            tot_dist += sum([x.data[1][:, 0].sum() for x in cur_seq])
        tot_n += (interval.end - interval.begin)
        tot_dist += interval.data[1][:, 0].sum()
        # Add time to de-conflict order
        heapq.heappush(PQ, (tot_dist, time.time(), (interval, cur_seq)))
    def enqueue_skip(begin, cur_seq):
        new_data = np.zeros((1, 2), dtype=np.float64)
        new_data[0, 0] = np.inf
        new_data[0, 1] = 100000
        new_data = (np.inf, new_data)
        enqueue_item(Interval(begin, begin+1, new_data), cur_seq.copy())

    for interval in tree[0]:
        enqueue_item(interval, [])
    if not len(PQ):
        enqueue_skip(0, [])

    #queue = deque([(interval, []) for interval in tree[0]])
    valid_seqs = []
    valid_dists = []
    longest_seq_t = -1
    longest_seqs = None
    n_expanded = 0
    #while not PQ.empty():
    while PQ:
        tot_dist, _, (interval, cur_seq) = heapq.heappop(PQ)
        n_expanded += 1
        if timeout > 0 and time.time() - start_time > timeout:
            break
        end = interval.end
        cur_seq.append(interval)
        # Greedy approach, only compute one valid seq? Or up to a certain number
        if end == T:
            valid_seqs.append(cur_seq)
            valid_dists.append(tot_dist)
            break
        if end > longest_seq_t:
            longest_seqs = [cur_seq]
            longest_seq_t = end
        elif end == longest_seq_t:
            longest_seqs.append(cur_seq)
        next_intervals = tree[end]
        lane_id = interval.data[0]
        found_continuation = False
        for possible_neighbor in next_intervals:
            found_continuation = True
            if lane_id == np.inf or possible_neighbor.data[0] not in lane_graph[lane_id]:
                current_pos = traj[end]
                current_vel = traj_vel[end]
                current_heading = traj_heading[end]
                angle_diff, possible_trans = get_angle(current_pos, current_heading, 
                                                       current_vel, int(possible_neighbor.data[0]))
            else:
                possible_trans = True

            if possible_trans:
                # TODO: enqueue_skip only if no possible transitions? i.e. put found_continuation = True here?
                new_data = (possible_neighbor.data[0], possible_neighbor.data[1][end - possible_neighbor.begin:])
                enqueue_item(Interval(end, possible_neighbor.end, new_data), cur_seq.copy())
        if not found_continuation:
            enqueue_skip(end, cur_seq.copy())
    
    # only possible fail is timeout?
    # if len(valid_seqs):
    #     status = f'VALID_{dist_threshold}'
    # elif closest_lanes_per_timestep[:, 0, 0].max() > dist_threshold/2:
    #     status = f'INVALID_{dist_threshold}_mindist'
    # else:
    #     status = f'INVALID_{dist_threshold}_meat'

    # Build return data
    ret = []
    unique_paths = set()
    for valid_seq in valid_seqs:
        seq_dists = []
        lane_path = []
        for interval in valid_seq:
            assert interval.end - interval.begin == len(interval.data[1]), 'Mismatch in interval data'
            seq_dists.extend(interval.data[1][:, 0])
            lane_path.extend([interval.data[0]] * (interval.end - interval.begin))
        if tuple(lane_path) not in unique_paths:
            unique_paths.add(tuple(lane_path))
            ret.append({'sequence': valid_seq, 'dists': np.array(seq_dists), 'path': lane_path})
    n_valid = 0
    n_tot = T
    if len(ret):
        lanes_used = set(ret[0]['path'])
        if np.inf not in lanes_used:
            status = f'VALID_{dist_threshold}_full'
        elif len(lanes_used) != 1:
            status = f'VALID_{dist_threshold}_partial'
        else:
            status = f'INVALID_{dist_threshold}'
        n_valid = (np.array(ret[0]['path']) != np.inf).sum()
    else:
            status = f'INVALID_{dist_threshold}'
    if 'full' not in status:
        status += '_meat' if closest_lanes_per_timestep[:, 0, 0].max() < dist_threshold/2 else '_mindist'

    return ret, status, n_expanded, time.time() - start_time, tree, n_valid, n_tot


def plot_traj(tag, static_map_infos, dynamic_map_infos, traj_xyz, agent_n, object_type, tree, best_seq):
    ac = 0
    fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 5 * 1))
    
    static_map_pos = plot_static_map_infos(static_map_infos, ax)
    # dynamic_map_pos = plot_dynamic_map_infos(dynamic_map_infos, ax)
    lane_pos = static_map_pos['lane']
    
    # Plot intervals
    for interval in tree:
        lane_idx = int(interval.data[0])
        lane = lane_pos[lane_idx]
        ax[ac].plot(lane[:, 0], lane[:, 1], color='green', linewidth=1, alpha=0.5)
    
    color = 'blue' if 'vehicle' in object_type.lower() else 'red'
    if best_seq['sequence'] is not None:
        intervals = best_seq['sequence']
        alpha = 1.0 / (len(intervals)+1)
        alpha_v = alpha
        for interval in intervals:
            lane_idx = int(interval.data[0])
            lane = static_map_pos['lane'][lane_idx]
            if 'vehicle' in object_type.lower():
                ax[ac].plot(lane[:, 0], lane[:, 1], c=cm.autumn(alpha_v), linewidth=2)
            elif 'cyclist' in object_type.lower():
                ax[ac].plot(lane[:, 0], lane[:, 1], c=cm.winter(alpha_v), linewidth=2)
            else:
                ax[ac].plot(lane[:, 0], lane[:, 1], c=cm.spring(alpha_v), linewidth=2)
            alpha_v += alpha

    ax[ac].plot(traj_xyz[:, 0], traj_xyz[:, 1], color=color, linewidth=1, linestyle='dashed')

    extent = ax[ac].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(VISUAL_OUT_SUBDIR, f"{tag}_{agent_n}_{object_type.lower()}.png"), dpi=300, bbox_inches=extent)
    plt.show()
    plt.close()


def process_file(path: str, meta: str,  closest_lanes = None, plot = False, tag: str ='temp', 
                 dist_threshold=10, prob_threshold=0.5, angle_threshold=20, thresh_iters=1,
                 hist_only=False, timeout=-1):

    # Load the scenario 
    with open(path, 'rb') as f:
        scenario = pkl.load(f)
    
    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    objects_type = track_infos['object_type']
    object_ids = track_infos['object_id']

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    dynamic_map_infos = scenario['dynamic_map_infos']
    # polylines = static_map_infos['all_polylines'][:, :3][:, None, :]

    static_map_pos = plot_static_map_infos(static_map_infos, ax=None, dim=3)
    lane_pos = static_map_pos['lane']
    lane_graph = build_lane_graph(static_map_infos['lane'])
    lane_segments = [np.stack([lane[:-1, :], lane[1:, :]], axis=1) for lane in lane_pos]
    
    last_t = 91 if not hist_only else 11
    # Trajectories --> (num_agents, time_steps, 9)
    trajectories = track_infos['trajs'][:, :last_t, :-1]
    # Mask         --> (num_agents, time_steps, 1)
    valid_masks = track_infos['trajs'][:, :last_t, -1] > 0

    num_agents, time_steps, dim = trajectories.shape
    
    meta_info = []

    best_seqs = []
    for n in range(num_agents):
        mask = valid_masks[n]
        if not np.any(mask):
            continue
        
        trajectory = trajectories[n, :, POS_XYZ_IDX]
        avg_speed = np.linalg.norm(trajectories[n, :, VEL_XY_IDX].T[mask], axis=-1).mean()
        # Avg speed < 1.0 m/s; where linear interpolation shouldn't matter that much compared to Frenet
        stationary = (avg_speed < 0.25)

        
        trajectory_closest_lanes = compute_k_closest_lanes(trajectory, mask, lane_pos) \
            if closest_lanes is None else closest_lanes[n]
        trajectory_closest_lanes = trajectory_closest_lanes[:, :, [0, -1]]
        
        thresholds = [dist_threshold*(2**i) for i in range(thresh_iters)]
        for threshold in thresholds:
            lane_sequences, status, n_expanded, tot_time, tree, n_valid, n_tot = build_lane_sequences(trajectory_closest_lanes[mask], lane_graph, lane_segments,
                                                threshold, prob_threshold, angle_threshold,
                                                trajectories[n, :, POS_XYZ_IDX].T[mask],
                                                trajectories[n, :, VEL_XY_IDX].T[mask],
                                                trajectories[n, :, HEADING_IDX].T[mask],
                                                track_infos, n, timeout)
            if len(lane_sequences) and n_valid == n_tot:
                break

        cur_info = {
            'obj_type': objects_type[n],
            'agent_n': n,
            'stationary': stationary,
            'scenario_id': scenario['scenario_id'],
            'object_id': object_ids[n],
            'valid': 'INVALID' not in status,
            'full': 'full' in status,
            'meat_threshold': threshold,
            'min_dist_okay': 'mindist' not in status,
            'avg_speed': avg_speed,
            'n_expanded': n_expanded,
            'tot_time': tot_time,
            'timeout': timeout > 0 and tot_time > timeout,
            'n_valid': n_valid,
            'n_tot': n_tot
        }
        meta_info.append(cur_info)

        if len(lane_sequences):
            seq_dists = [x['dists'].mean() for x in lane_sequences]
            best_seq_idx = np.argmin(seq_dists)
            best_seq = lane_sequences[best_seq_idx]
            best_seq['closest_lanes'] = trajectory_closest_lanes[mask]
            best_seqs.append(best_seq)
        else:
            best_seq = {'sequence': None, 'dists': None, 'path': None,
                        'closest_lanes': trajectory_closest_lanes[mask]}
            best_seqs.append(best_seq)
        # if not cur_info['stationary'] and cur_info['valid']:
        # if not cur_info['stationary'] and not cur_info['valid'] and cur_info['min_dist_okay']:
        #     traj_xyz = trajectories[n, :, POS_XYZ_IDX].T[mask]
        #     plot_traj(tag, static_map_infos, dynamic_map_infos, traj_xyz, n, objects_type[n], tree, best_seq)
        
        # mtraj = trajectory.T[mask]
        # motion = np.linalg.norm(mtraj[1:] - mtraj[:-1], axis=1).sum()
        # valseq0 = lane_sequences['valid_sequences'][0] 
    return meta_info, best_seqs
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument('--cache_path', type=str, default='/av_shared_ssd/scenario_id/cache')
    parser.add_argument(
        '--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--dist_threshold', type=float, default=5)
    parser.add_argument('--angle_threshold', type=float, default=45)
    parser.add_argument('--thresh_iters', type=int, default=1)
    parser.add_argument('--hist_only', action='store_true')
    # parser.add_argument('--greedy_factor', type=int, default=1)
    # parser.add_argument('--greedy_iters', type=int, default=1)
    parser.add_argument('--prob_threshold', type=float, default=0.5)
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--load_cache', action='store_true')
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()

    # Uses the "new" splits, from resplit.py; this way, test is labeled as well
    base = os.path.expanduser(args.base_path)
    step = 1
    if args.split == 'training':
        step = 5
        scenarios_base = f'{base}/new_processed_scenarios_training'
        scenarios_meta = f'{base}/new_processed_scenarios_training_infos.pkl'
    elif args.split == 'validation':
        scenarios_base = f'{base}/new_processed_scenarios_validation'
        scenarios_meta = f'{base}/new_processed_scenarios_val_infos.pkl'
    else:
        scenarios_base = f'{base}/new_processed_scenarios_testing'
        scenarios_meta = f'{base}/new_processed_scenarios_test_infos.pkl'

    start = time.time()
    if args.load_cache:
        print("Loading cache")
        closest_lanes_filepath = os.path.join(args.cache_path, f"{args.split}/closest_lanes/closest.npz")
        closest_lanes = np.load(closest_lanes_filepath, allow_pickle=True)['arr_0']
        print(f"Loading closest lanes took {time.time() - start} seconds")
    CACHE_SUBDIR = os.path.join(CACHE_DIR, args.split, 'meat_lanes')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)

    print(f"Loading Scenario Data...")
    with open(scenarios_meta, 'rb') as f:
        metas = pkl.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]
    
    if args.shards > 1:
        n_per_shard = np.ceil(len(metas)/args.shards)
        shard_start = int(n_per_shard*args.shard_idx)
        shard_end = int(n_per_shard*(args.shard_idx + 1))
        metas = metas[shard_start:shard_end]
        inputs = inputs[shard_start:shard_end]
    
    num_scenarios = len(metas)
    if args.num_scenarios != -1:
        num_scenarios = args.num_scenarios
        metas = metas[:num_scenarios]
        inputs = inputs[:num_scenarios]

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(delayed(process_file)(
            path, meta, closest_lanes[i] if args.load_cache else None, args.plot, 
            tag=f"{s.split('.')[0]}", dist_threshold=args.dist_threshold, prob_threshold=args.prob_threshold,
            angle_threshold=args.angle_threshold,
            thresh_iters = args.thresh_iters, hist_only=args.hist_only,
            timeout = args.timeout)
            #for i, ((s, path), meta) in tqdm(enumerate(zip(inputs, metas)), desc=msg, total=len(metas), disable=True))
            for i, ((s, path), meta) in tqdm(enumerate(zip(inputs, metas)), desc=msg, total=len(metas)))
    else:
        all_outs = []
        #for i, ((s, path), meta) in tqdm(enumerate(zip(inputs, metas)), msg, total=len(metas), disable=True):
        for i, ((s, path), meta) in tqdm(enumerate(zip(inputs, metas)), msg, total=len(metas)):
            #print(f'PROCESSING {i}: {path}')
            out = process_file(
                path, meta, closest_lanes[i] if args.load_cache else None, args.plot, 
                tag=f"{s.split('.')[0]}", dist_threshold=args.dist_threshold, prob_threshold=args.prob_threshold,
                angle_threshold=args.angle_threshold,
                thresh_iters = args.thresh_iters, hist_only=args.hist_only,
                timeout = args.timeout)
            all_outs.append(out)

    best_seqs = [out[-1] for out in all_outs]
    all_dists = [x['dists'].mean() for best_seq in best_seqs for x in best_seq if x['dists'] is not None]
    all_meta = pd.DataFrame([x for out in all_outs for x in out[0]])
    all_meta['valid_rate'] = all_meta['n_valid']/all_meta['n_tot']
    print(f"Process took {time.time() - start} seconds.")

    print(f'Saving {len(all_outs)} scenarios...')
    # with open(f'{CACHE_SUBDIR}/meat_lanes.pkl', 'wb') as f:
    #     pkl.dump(all_outs, f)
    shard_suffix = f'_shard{args.shard_idx}_{args.shards}' if args.shards > 1 else ''
    if not args.hist_only:
        with open(f'{CACHE_SUBDIR}/lanes{shard_suffix}.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
        all_meta.to_csv(f'{CACHE_SUBDIR}/lanes_meta{shard_suffix}.csv')
    else:
        with open(f'{CACHE_SUBDIR}/lanes_hist{shard_suffix}.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
        all_meta.to_csv(f'{CACHE_SUBDIR}/lanes_hist_meta{shard_suffix}.csv')
    
    print("Done.")
