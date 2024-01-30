import itertools
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
from matplotlib import cm


from utils.common import (
    get_agent_dims, compute_leading_agent, compute_dists_to_conflict_points, is_sharing_lane, 
    is_increasing, IPOS_XY_IDX, IPOS_SD_IDX, ILANE_IDX, IVALID_IDX, AGENT_NUM_TO_TYPE
)
from utils.visualization import plot_static_map_infos, plot_dynamic_map_infos, plot_interaction

class Status(Enum):
    UNKNOWN = -1
    OK = 0
    NOK_MASK = 1
    NOK_SEPARATION = 2
    NOK_LANE = 3
    NOK_STATIONARY = 4
    NOK_NAN = 5
    NOK_HEADING = 6
    NOK_CDIST = 7
    NOK_AGENT_TYPE = 8
    NOK_NO_CPTS = 9

# --------------------------------------------------------------------------------------------------
# Feature Wrapper
# --------------------------------------------------------------------------------------------------
def compute_interaction_features(
    scenario: dict, conflict_points: np.array, load_type: str, 
    interp_trajectories: np.array = None, interp_velocities: np.array = None, 
    other_interp_trajectories: np.array = None, other_interp_velocities: np.array = None, 
    plot: bool = False, 
    tag: str = 'temp', timesteps: int = 91, hist_len: int = 11, hz: float = 10, 
    agent_scene_conflict_dist_thresh: float = 5.0, agent_agent_conflict_dist_thresh: float = 1.0, 
    agent_agent_dist_threshold: float = 50.0, speed_threshold: float = 0.25, 
    heading_threshold: float = 45.0,
    cluster_anomaly: np.array = None
):
    """ Computes the interaction state for all pairs of agents. """
    supported_metrics = ['thw', 'ttc', 'scene_mttcp', 'agent_mttcp', 'drac', 'collisions', 'traj_pair_anomaly']

    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    trajectories = track_infos['trajs'][:, :, :-1]
    valid_masks = track_infos['trajs'][:, :, -1] > 0
    object_types = track_infos['object_type']

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    dynamic_map_infos = scenario['dynamic_map_infos']

    num_agents, _, _ = trajectories.shape
    agent_dims = get_agent_dims(trajectories, valid_masks)

    # conflict points ['static', 'dynamic', 'lane_intersections'] -> (N, 2)
    conflict_points_list = [] 
    cpts = None
    for k, v in conflict_points.items():
        if len(v) > 0:
            conflict_points_list.append(v[:, :2])
    if len(conflict_points_list) > 0:
        cpts = np.concatenate(conflict_points_list)

    if load_type != 'asym':
        dists_to_conflict_points = None
        if not cpts is None:
            dists_to_conflict_points = compute_dists_to_conflict_points(
                cpts, interp_trajectories, load_type)
        other_dists_to_conflict_points = None
    else:
        assert len(interp_trajectories) == len(other_interp_trajectories), 'Mismatch in gt/fe interp n agents'
        dists_to_conflict_points = None
        if not cpts is None:
            dists_to_conflict_points = compute_dists_to_conflict_points(
                cpts, interp_trajectories, load_type='gt')
        other_dists_to_conflict_points = None
        if not cpts is None:
            other_dists_to_conflict_points = compute_dists_to_conflict_points(
                cpts, other_interp_trajectories, load_type='fe')
    

    # Filter fully invalid agents first
    # all_agents = np.arange(num_agents)
    # n_valids = np.sum(interp_trajectories[:, :, -1], axis=-1)
    # all_agents = all_agents[n_valids > 0]
    if load_type != 'asym':
        agent_combinations = list(itertools.combinations(range(num_agents), 2))
    else:
        # Due to the assymetry
        agent_combinations = list(itertools.permutations(range(num_agents), 2))
    #to_valid_map = {i: new_idx for new_idx, i in enumerate(all_agents)}

    sinit = np.asarray([Status.UNKNOWN for _ in agent_combinations])

    state =  {
        "status": sinit.copy(), # Agent-Pair Status
        "thw": [],              # Time Headway
        "ttc": [],              # Time to Collision
        "scene_mttcp": [],      # Time-to scene-related conflict a point
        "agent_mttcp": [],      # Agent-to-agent time-to conflict a point
        "drac": [],             # Deceleration Rate to Avoid a Crash
        "collisions": [],       # Collisions
        "agent_ids": [(i, j) for i, j in agent_combinations],
        "agent_types": [(object_types[i], object_types[j]) for i, j in agent_combinations],
        "traj_pair_anomaly": []
    }
    

    for n, (i, j) in enumerate(agent_combinations):
        for metric in supported_metrics:
            state[metric].append([])
        state['traj_pair_anomaly'][-1] = 0
        
        # Skip pedestrian-pedestrian interactions 
        object_type_i, object_type_j = object_types[i], object_types[j]
        # if object_type_i == 'TYPE_PEDESTRIAN' and object_type_j == 'TYPE_PEDESTRIAN':
        #     state['status'][n] = Status.NOK_AGENT_TYPE
        #     continue
        
        # Skip nan trajectories
        mask = None
        time = np.arange(0, timesteps / hz, 1 / hz)

        # Assume that pos_i will be the one with FE and pos_j will be GT
        if load_type != 'asym':
            pos_i = interp_trajectories[i, :, IPOS_XY_IDX].T
            pos_j = interp_trajectories[j, :, IPOS_XY_IDX].T
            frenet_i = interp_trajectories[i, :, IPOS_SD_IDX].T
            frenet_j = interp_trajectories[j, :, IPOS_SD_IDX].T
            lane_i = interp_trajectories[i, :, ILANE_IDX].T
            lane_j = interp_trajectories[j, :, ILANE_IDX].T
        else:
            pos_i = other_interp_trajectories[i, :, IPOS_XY_IDX].T
            pos_j = interp_trajectories[j, :, IPOS_XY_IDX].T
            frenet_i = other_interp_trajectories[i, :, IPOS_SD_IDX].T
            frenet_j = interp_trajectories[j, :, IPOS_SD_IDX].T
            lane_i = other_interp_trajectories[i, :, ILANE_IDX].T
            lane_j = interp_trajectories[j, :, ILANE_IDX].T
        
        if np.all(np.isnan(pos_i)) or np.all(np.isnan(pos_j)):
            state['status'][n] = Status.NOK_NAN
            continue
        
        len_i, len_j = agent_dims[i, 0], agent_dims[j, 0]

        if load_type != 'asym':
            vel_i, vel_j = interp_velocities[i], interp_velocities[j]
        else:
            vel_i, vel_j = other_interp_velocities[i], interp_velocities[j]

        heading_i = np.rad2deg(np.arctan2(vel_i[:, 1], vel_i[:, 0]))
        heading_j = np.rad2deg(np.arctan2(vel_j[:, 1], vel_j[:, 0]))
        dist_cp_i, dist_cp_j = None, None
        if load_type != 'asym':
            if not cpts is None:
                dist_cp_i, dist_cp_j = dists_to_conflict_points[i], dists_to_conflict_points[j]
        else:
            if not cpts is None:
                dist_cp_i, dist_cp_j = other_dists_to_conflict_points[i], dists_to_conflict_points[j]

        # If using interp but not extrap, mask the valid region of the trajectories. 
        if load_type == 'gt' or load_type == 'asym':
            mask_j = interp_trajectories[j, :, IVALID_IDX].T.astype(int)
            if load_type != 'asym':
                mask_i = interp_trajectories[i, :, IVALID_IDX].T.astype(int)
            else:
                # Since FE, it's all valid
                mask_i = np.ones_like(mask_j)
            mask = np.where(mask_i & mask_j == 1)[0]
            if mask.shape[0] < 1:
                state['status'][n] = Status.NOK_MASK
                continue

            start, end = mask[0], mask[-1]+1
            
            time = time[start:end]
            pos_i, pos_j = pos_i[start:end], pos_j[start:end]
            vel_i, vel_j = vel_i[start:end], vel_j[start:end]
            lane_i, lane_j = lane_i[start:end], lane_j[start:end]
            frenet_i, frenet_j = frenet_i[start:end], frenet_j[start:end]
            heading_i, heading_j = heading_i[start:end], heading_j[start:end]
            
            if not cpts is None:
                dist_cp_i, dist_cp_j = dist_cp_i[:, start:end], dist_cp_j[:, start:end]  
            

        # For now, stick to cluster_anomaly between GT trajs as a starting point
        anomaly_dists_i, anomaly_ids_i = cluster_anomaly['pairs'][i], cluster_anomaly['pair_ids'][i]
        anomaly_dists_j, anomaly_ids_j = cluster_anomaly['pairs'][j], cluster_anomaly['pair_ids'][j]
        anomaly_mask_i = anomaly_ids_i == j
        anomaly_mask_j = anomaly_ids_j == i
        anomaly_dist_i = anomaly_dists_i[anomaly_mask_i].sum()
        anomaly_dist_j = anomaly_dists_j[anomaly_mask_j].sum()
        state['traj_pair_anomaly'][n] = max(anomaly_dist_i, anomaly_dist_j)

        # Skip if both agents are stationary
        v_i = np.linalg.norm(vel_i, axis=-1) 
        v_j = np.linalg.norm(vel_j, axis=-1)
        is_stationary_i = v_i.mean() <= speed_threshold
        is_stationary_j = v_j.mean() <= speed_threshold
        if is_stationary_i and is_stationary_j:
            state['status'][n] = Status.NOK_STATIONARY
            continue

        # Skip if agents are not within a distance threhold from each other
        d_ij = np.linalg.norm(pos_i - pos_j, axis=1)
        if not np.any(d_ij < agent_agent_dist_threshold):
            state['status'][n] = Status.NOK_CDIST
            continue

        # Skip if the distance between the agents is increasing over time
        # if is_increasing(d_ij, time): 
        #     state['status'][n] = Status.NOK_SEPARATION
        #     continue
        
        in_cp_i, in_cp_j = 0, 0
        if not cpts is None:
            in_cp_i = (dist_cp_i < agent_scene_conflict_dist_thresh).sum(axis=1)
            in_cp_j = (dist_cp_j < agent_scene_conflict_dist_thresh).sum(axis=1)

        agent_i = (pos_i, vel_i, heading_i, len_i, object_type_i, is_stationary_i, in_cp_i, dist_cp_i)
        agent_j = (pos_j, vel_j, heading_j, len_j, object_type_j, is_stationary_j, in_cp_j, dist_cp_j)

        # ------------------------------------------------------------------------------------------
        # Compute mTTCP for conflict points in the scene. I divided mTTCP into two metrics:
        #     * Scence mTTCP: Scene conflict points are points obtained from the following regions:
        #                     crosswalk, speed bump, traffic light, stop signs and approximate
        #                     lane intersections. 
        #
        #     * Agent mTTCP: I also consider if given two agent trajectories pass through the same
        #                    point. Once identified, I calculate the mTTCP from t=0 to t=first time
        #                    one of the agents cross that conflict point. This (I think), is aligned
        #                    with what's done in the ExiD paper (same PI as the analysis framework:
        #                    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9827305)
        # ------------------------------------------------------------------------------------------
        scene_mttcp = np.inf * np.ones(pos_i.shape[0])
        if not cpts is None:
            scene_mttcp, scene_min_t, scene_shared_cps = compute_scene_mttcp(agent_i, agent_j, time)
        state['scene_mttcp'][n] = scene_mttcp

        agent_mttcp, agent_min_t, agent_shared_cps = compute_agent_mttcp(
            agent_i, agent_j, time, agent_agent_conflict_dist_thresh)
        state['agent_mttcp'][n] = agent_mttcp

        collisions = compute_collisions(agent_i, agent_j)
        state['collisions'][n] = collisions

        # ------------------------------------------------------------------------------------------
        # Compute THW, TTC, DRAC, which currently, all depend on whether the agents share a lane
        # at some point in their trajectories. We don't need to make this assumption but I think it
        # simplifies handling cases. Also, these metrics are usually considered for agents sharing 
        # a lane and for simple cases where the leader/follower are clearly and easily identified. 
        # I currently identify the leading agent based on the relative heading between the agents 
        # and their positions within the map. I'm still unsure what's the best way to deal with this.
        # ------------------------------------------------------------------------------------------

        # For the rest of the metrics, the agents must be sharing a lane along their sequences. 
        if not is_sharing_lane(lane_i, lane_j):
            state['status'][n] = Status.NOK_LANE
            continue        

        # Determine the leading agent. First, check if the agents' headings are within a valid 
        # threshold. Then use the valid headings to determine who's leading at each step based on 
        # their positions on the scene.
        heading_diff = np.abs((heading_j - heading_i + 540) % 360 - 180)
        valid_heading = np.where(heading_diff < heading_threshold)[0]
        if valid_heading.shape[0] < 2:
            state['status'][n] = Status.NOK_HEADING
            continue
        

        # At this point, guaranteed to be sharing a lane for a certain part
        leading_agent = compute_leading_agent(agent_i, agent_j, valid_heading)

        thw, ttc, drac = compute_interaction_state(
            agent_i, agent_j, leading_agent, time, d_ij, valid_heading)
        
        state['thw'][n] = thw
        state['ttc'][n] = ttc
        state['drac'][n] = drac

        # All the metrics were computed
        state['status'][n] = Status.OK if not cpts is None else Status.NOK_NO_CPTS

        # all_valid = (
        #     ttc.min() < float('inf') and thw.min() < float('inf') and drac.max() > 0.0 and 
        #     scene_mttcp.min() < float('inf') and agent_mttcp.min() < float('inf'))
        
        # if plot and all_valid:
        #     num_windows = 6
        #     point_size = 1
        #     alpha = 0.5
        #     fig, ax = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        #     static_map_infos = scenario['map_infos']
        #     dynamic_map_infos = scenario['dynamic_map_infos']

        #     plot_static_map_infos(static_map_infos, ax)
        #     plot_dynamic_map_infos(dynamic_map_infos, ax)

        #     for na in range(num_agents):
                
        #         color = 'red' if na == i or na == j else 'blue'

        #         # Uses interpolated and extrapolated regions 
        #         if np.any(np.isnan(interp_trajectories[na, :, IPOS_XY_IDX])):
        #             continue
        #         pos = interp_trajectories[na, :, IPOS_XY_IDX].T

        #         if load_type == 'gt':
        #             # Uses interpolated regions only
        #             mask = np.where(interp_trajectories[na, :, -1] == 1)[0]
        #             if mask.shape[0] == 0:
        #                 continue
        #             pos = interp_trajectories[na, mask[0]:(mask[-1]+1), IPOS_XY_IDX].T
                
        #         ax[0].plot(pos[:, 0], pos[:, 1], color=color)
        #         ax[0].set_title('Full Scene')

        #     for i in range(1, num_windows):
        #         ax[i].scatter(pos_i[0, 0], pos_i[0, 1], color='blue', s=3, zorder=10)
        #         ax[i].scatter(pos_j[0, 0], pos_j[0, 1], color='blue', s=3, zorder=10)
        #         ax[i].plot(pos_i[:, 0], pos_i[:, 1], color='blue', alpha=0.2)
        #         ax[i].plot(pos_j[:, 0], pos_j[:, 1], color='blue', alpha=0.2)
    
        #     size = 1

        #     # THW
        #     thw_min = valid_heading[np.argmin(thw)]
        #     ax[1].scatter(pos_i[thw_min, 0], pos_i[thw_min, 1], s=size+2, c='red', zorder=10)
        #     ax[1].scatter(pos_j[thw_min, 0], pos_j[thw_min, 1], s=size+2, c='red', zorder=10)
        #     ax[1].scatter(pos_i[valid_heading, 0], pos_i[valid_heading, 1], s=size, c=cm.winter(thw))
        #     ax[1].scatter(pos_j[valid_heading, 0], pos_j[valid_heading, 1], s=size, c=cm.winter(thw))
        #     ax[1].set_title(f'Min THW: {round(thw.min(), 3)}')

        #     # TTC
        #     ttc_min = valid_heading[np.argmin(ttc)]
        #     ax[2].scatter(pos_i[ttc_min, 0], pos_i[ttc_min, 1], s=size+2, c='red', zorder=10)
        #     ax[2].scatter(pos_j[ttc_min, 0], pos_j[ttc_min, 1], s=size+2, c='red', zorder=10)
        #     ax[2].scatter(pos_i[valid_heading, 0], pos_i[valid_heading, 1], s=size, c=cm.winter(ttc))
        #     ax[2].scatter(pos_j[valid_heading, 0], pos_j[valid_heading, 1], s=size, c=cm.winter(ttc))
        #     ax[2].set_title(f'Min TTC: {round(ttc.min(), 3)}')

        #     # DRAC
        #     drac_max = valid_heading[np.argmax(drac)]
        #     ax[3].scatter(pos_i[drac_max, 0], pos_i[drac_max, 1], s=size+2, c='red', zorder=10)
        #     ax[3].scatter(pos_j[drac_max, 0], pos_j[drac_max, 1], s=size+2, c='red', zorder=10)
        #     ax[3].scatter(pos_i[valid_heading, 0], pos_i[valid_heading, 1], s=size, c=cm.winter(drac))
        #     ax[3].scatter(pos_j[valid_heading, 0], pos_j[valid_heading, 1], s=size, c=cm.winter(drac))
        #     ax[3].set_title(f'Max DRAC: {round(drac.max(), 3)}')

        #     # Scene mTTCP
        #     t_min = scene_mttcp.argmin()
        #     scene_mttcp_ = 1 / scene_mttcp
        #     scene_mttcp_ /= scene_mttcp_.max()
        #     for n in range(scene_shared_cps.shape[0]):
        #         color = 'purple' if t_min == n else 'red'
        #         scp = scene_shared_cps[n]
        #         t = int(scene_min_t[n])
        #         ax[4].scatter(conflict_points[scp, 0], conflict_points[scp, 1], s=size, color='orange')
        #         ax[4].scatter(pos_i[t, 0], pos_i[t, 1], s=size, color=color, alpha=scene_mttcp_[n])
        #         ax[4].scatter(pos_j[t, 0], pos_j[t, 1], s=size, color=color, alpha=scene_mttcp_[n])
        #     ax[4].set_title(f'Scene mTTCP {round(scene_mttcp.min(), 3)}')

        #     # Agent mTTCP
        #     t_min = agent_mttcp.argmin()
        #     agent_mttcp_ = 1 / agent_mttcp
        #     agent_mttcp_ /= agent_mttcp_.max()
        #     ax[5].scatter(agent_shared_cps[:, 0], agent_shared_cps[:, 1], s=size, color='orange')
        #     for n in range(agent_shared_cps.shape[0]):
        #         color = 'purple' if t_min == n else 'red'
        #         t = int(agent_min_t[n])
        #         ax[5].scatter(pos_i[t, 0], pos_i[t, 1], s=size, color=color, alpha=agent_mttcp_[n])
        #         ax[5].scatter(pos_j[t, 0], pos_j[t, 1], s=size, color=color, alpha=agent_mttcp_[n])
        #     ax[5].set_title(f'Agent mTTCP {round(agent_mttcp.min(), 3)}')

        #     for a in ax.reshape(-1):
        #         a.set_xticks([])
        #         a.set_yticks([])
            
        #     plt.subplots_adjust(wspace=0.05)
        #     plt.suptitle(f'Interaction of agents {i} and {j}')
        #     plt.savefig(
        #         f"{tag}_{i}-{j}_interaction_thw{round(thw.min(), 3)}_ttc{round(ttc.min())}_drac{round(drac.max())}.png", 
        #         dpi=300, bbox_inches='tight')
        #     plt.show()
        #     plt.close()
        
    return state


# --------------------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------------------
def compute_agent_mttcp(agent_i, agent_j, time, dist_threshold: float = 1.0, eps = 0.001):
    pos_i, vel_i, heading_i, len_i, agent_type_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, len_j, agent_type_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    # T, 2 -> T, T
    dists = np.linalg.norm(pos_i[:, None, :] - pos_j, axis=-1)
    i_idx, j_idx = np.where(dists <= dist_threshold)
    
    if len(i_idx) == 0:
        return np.inf * np.ones(1), None, None
    
    vals, i_unique = np.unique(i_idx, return_index=True)
    i_idx, j_idx = i_idx[i_unique], j_idx[i_unique]

    agent_conflict_points = pos_i[i_idx]    
    agents_mttcp = np.inf * np.ones(agent_conflict_points.shape[0])
    min_ts = np.inf * np.ones(agent_conflict_points.shape[0])

    v_i = np.linalg.norm(vel_i, axis=-1) + eps
    v_j = np.linalg.norm(vel_j, axis=-1) + eps

    # Agent i is already at the conflict point
    if is_stationary_i:
        t = j_idx[0] + 1
        agents_mttcp = np.abs(dists[0, :t] / v_j[:t]).min()
        min_ts = agents_mttcp.argmin()

    # Agent j is already at the conflict point
    elif is_stationary_j:
        t = i_idx[0] + 1
        agents_mttcp = np.abs(dists[:t, 0] / v_i[:t]).min()
        min_ts = agents_mttcp.argmin()
    
    # Agents are moving
    else:
        for n, (i, j) in enumerate(zip(i_idx, j_idx)):
            conflict_point = pos_i[i] # which should be == pos_j[j] 
            t = min(i, j) + 1
            ttcp_i = np.linalg.norm(conflict_point - pos_i[:t], axis=-1) / v_i[:t]
            ttcp_j = np.linalg.norm(conflict_point - pos_j[:t], axis=-1) / v_j[:t]
            ttcp = np.abs(ttcp_i - ttcp_j)
            agents_mttcp[n] = ttcp.min()
            min_ts[n] = ttcp.argmin()

    return agents_mttcp, min_ts, agent_conflict_points

def compute_scene_mttcp(agent_i, agent_j, time, eps = 0.001):
    pos_i, vel_i, heading_i, len_i, agent_type_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, len_j, agent_type_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    if len(in_cp_i) == 0 and len(in_cp_j) == 0:    
        return np.inf * np.ones(1), None, None

    idx_in_cp_i, idx_in_cp_j = np.where(in_cp_i > 0)[0], np.where(in_cp_j > 0)[0]
    shared_conflict_points = np.intersect1d(idx_in_cp_i, idx_in_cp_j)
    if shared_conflict_points.shape[0] == 0:
        return np.inf * np.ones(1), None, None

    v_i = np.linalg.norm(vel_i, axis=-1) + eps
    v_j = np.linalg.norm(vel_j, axis=-1) + eps

    # TODO: check if both velocities are approximately facing the conflict point?

    # Agent i is already at the conflict point
    scene_mttcp = np.inf * np.ones(shared_conflict_points.shape[0])
    min_ts = np.inf * np.ones(shared_conflict_points.shape[0])
    if is_stationary_i:
        for n in range(shared_conflict_points.shape[0]):
            scp = shared_conflict_points[n]
            min_t = dist_cp_j[scp].argmin() + 1
            ttcp_j = dist_cp_j[scp, :min_t] / v_j[:min_t]
            scene_mttcp[n] = np.abs(ttcp_j).min()
            min_ts[n] = np.abs(ttcp_j).argmin()

    # Agent j is already at the conlict point 
    elif is_stationary_j:
        for n in range(shared_conflict_points.shape[0]):
            scp = shared_conflict_points[n]
            min_t = dist_cp_i[scp].argmin() + 1
            ttcp_i = dist_cp_i[scp, :min_t] / v_i[:min_t]
            scene_mttcp[n] = np.abs(ttcp_i).min()
            min_ts[n] = np.abs(ttcp_i).argmin()
    
    # Agents are moving
    else:
        for n in range(shared_conflict_points.shape[0]):
            scp = shared_conflict_points[n]
            # should this be max or min here?
            # also, should we really filter independetly min_t?
            # do we need to ensure that both people are actually getting closer to the conflict point?
            min_t = min(dist_cp_i[scp].argmin(), dist_cp_j[scp].argmin()) + 1
            ttcp_i = dist_cp_i[scp, :min_t] / v_i[:min_t]
            ttcp_j = dist_cp_j[scp, :min_t] / v_j[:min_t]
            scene_mttcp[n] = np.abs(ttcp_i - ttcp_j).min()
            min_ts[n] = np.abs(ttcp_i - ttcp_j).argmin()

    return scene_mttcp, min_ts, shared_conflict_points

def compute_interaction_state(
    agent_i, agent_j, leading_agent, time, dists_ij, heading_mask = None, eps = 0.0001
):
    """ Computes the following measurements:
    
        Time Headway (THW):
        -------------------
            TWH = t_i - t_j
        where t_i is the time vehicle passes a certain location and t_j is the time the vehicle ahead 
        passes that same location. 

        Time-to-Collision (TTC):
        ------------------------
                  x_j - x_i - l_i
            TTC = ---------------  forall v_i > v_j
                     v_i - v_j

        where x_i, l_i, and v_i are the position, length and speed of the following vehicle and 
        x_j and v_j are the position and speed of the leading vehicle.

        Deceleration Rate to Avoid a Crash (DRAC): 
        -----------------------------------------
                    (v_j - v_i) ** 2
            DRAC = ------------------  
                      2 (x_i - x_j)
        the average delay of a road user to avoid an accident at given velocities and distance 
        between vehicles, where i is the leader and j is the follower. 
    """
    pos_i, vel_i, heading_i, len_i, agent_type_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, len_j, agent_type_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j

    if not heading_mask is None:
        pos_i, pos_j = pos_i[heading_mask], pos_j[heading_mask]
        heading_i, heading_j = heading_i[heading_mask], heading_j[heading_mask]
        time = time[heading_mask]
    
    v_i = np.linalg.norm(vel_i, axis=-1) + eps
    v_j = np.linalg.norm(vel_j, axis=-1) + eps

    thw = np.inf * np.ones(shape=(pos_i.shape[0]))
    ttc = np.inf * np.ones(shape=(pos_i.shape[0]))
    drac = np.zeros(shape=(pos_i.shape[0]))

    pos_len_i, pos_len_j = np.zeros_like(pos_i), np.zeros_like(pos_j)
    pos_len_i[:, 0] = pos_i[:, 0] + len_i * np.cos(np.deg2rad(heading_i))
    pos_len_i[:, 1] = pos_i[:, 1] + len_i * np.sin(np.deg2rad(heading_i))

    pos_len_j[:, 0] = pos_j[:, 0] + len_j * np.cos(np.deg2rad(heading_j))
    pos_len_j[:, 1] = pos_j[:, 1] + len_j * np.sin(np.deg2rad(heading_j))

    # ...where i is the agent ahead
    i_idx = np.where(leading_agent == 0)[0]
    # ...where j is the agent ahead
    j_idx = np.where(leading_agent == 1)[0]
    # ...where i is the leader but j's speed is higher
    v_i_idx = np.intersect1d(i_idx, np.where(v_j > v_i)[0])
    # ...where j is the leader but i's speed is higher
    v_j_idx = np.intersect1d(j_idx, np.where(v_i > v_j)[0])

    # THW
    t_j = dists_ij / v_j
    t_i = dists_ij / v_i
    
    # TTC
    dpos_ij = np.linalg.norm(pos_i[v_i_idx] - pos_len_j[v_i_idx], axis=-1)
    dpos_ji = np.linalg.norm(pos_j[v_j_idx] - pos_len_i[v_j_idx], axis=-1)

    if is_stationary_i:
        thw[i_idx] = t_j[i_idx] 

        ttc[v_i_idx] = dpos_ij / v_j[v_i_idx] 

    elif is_stationary_j:
        thw[j_idx]  = t_i[j_idx]
        
        ttc[v_j_idx] = dpos_ji / v_i[v_j_idx]

    else:
        # TODO: why subtracting off time[i_idx] and time[j_idx] respectively? 
        # when t_j and t_i are already computed at the *same* timesteps
        thw[i_idx] = t_j[i_idx]# - time[i_idx]
        thw[j_idx] = t_i[j_idx]# - time[j_idx]
        
        ttc[v_i_idx] = dpos_ij / (v_j[v_i_idx] - v_i[v_i_idx])
        ttc[v_j_idx] = dpos_ji / (v_i[v_j_idx] - v_j[v_j_idx])
    
    dpos_ij = np.linalg.norm(pos_i[v_i_idx] - pos_j[v_i_idx], axis=-1)
    drac[v_i_idx] = ((v_j[v_i_idx] - v_i[v_i_idx]) ** 2) / (2 * dpos_ij)

    dpos_ji = np.linalg.norm(pos_j[v_j_idx] - pos_i[v_j_idx], axis=-1)
    drac[v_j_idx] = ((v_i[v_j_idx] - v_j[v_j_idx]) ** 2) / (2 * dpos_ji)

    return thw, ttc, drac

def compute_collisions(agent_i, agent_j, collision_threshold: float = 0.25):
    pos_i, vel_i, heading_i, len_i, agent_type_i, is_stationary_i, in_cp_i, dist_cp_i = agent_i
    pos_j, vel_j, heading_j, len_j, agent_type_j, is_stationary_j, in_cp_j, dist_cp_j = agent_j
    # TODO: update to handle segment change and such
    from shapely import LineString
    segments_i = np.stack([pos_i[:-1], pos_i[1:]], axis=1)
    segments_j = np.stack([pos_j[:-1], pos_j[1:]], axis=1)
    segments_i = [LineString(x) for x in segments_i]
    segments_j = [LineString(x) for x in segments_j]
    # TODO: update to *also* count collisions where distance between center points of people is less than threshold
    # Maybe just return value of how close they get, to assign a threshold later?
    thresh_res = np.linalg.norm(pos_i - pos_j, axis=-1) <= collision_threshold
    return (np.array([False] + [x.intersects(y) for x, y in zip(segments_i, segments_j)])) | thresh_res

# --------------------------------------------------------------------------------------------------
# Score Wrapper. Currently, just used for testing it computes features and scores but ideally should 
# take already computed features and get scores out of it. 
# --------------------------------------------------------------------------------------------------
def compute_interaction_scores(
    scenario: dict, conflict_points: np.array, closest_lanes: np.array, use_interp: bool = False, 
    use_extrap: bool = False, hist_only: bool = False, interp_trajectories: np.array = None,
    interp_velocities: np.array = None, plot: bool = False, tag: str = 'temp', timesteps: int = 91,
    hist_len: int = 11, hz: float = 10, agent_scene_conflict_dist_thresh: float = 5.0, 
    agent_agent_conflict_dist_thresh: float = 1.0, agent_agent_dist_threshold: float = 100.0, 
    speed_threshold: float = 0.25, heading_threshold: float = 45.0, time_threshold: float = 5.0,
    deceleration_threshold: float = 10.0
):
    raise NotImplementedError
    """ Computes the interaction state for all pairs of agents. """
    supported_metrics = ['thw', 'ttc', 'scene_mttcp', 'agent_mttcp', 'drac']

    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    trajectories = track_infos['trajs'][:, :, :-1]
    valid_masks = track_infos['trajs'][:, :, -1] > 0
    object_types = track_infos['object_type']

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    dynamic_map_infos = scenario['dynamic_map_infos']

    num_agents, _, _ = trajectories.shape
    agent_dims = get_agent_dims(trajectories, valid_masks)
    agent_combinations = list(itertools.combinations(range(num_agents), 2))

    sinit = np.asarray([Status.UNKNOWN for _ in agent_combinations])

    state =  {
        "status": sinit.copy(),
        # Time Headway
        "thw": {"values": [], "detections": [], "scores": []},
        # Time to Collision
        "ttc": {"values": [], "detections": [], "scores": []},
        # Time-to scene-related conflict a point
        "scene_mttcp": {"values": [], "detections": [], "scores": []},
        # Agent-to-agent time-to conflict a point
        "agent_mttcp": {"values": [], "detections": [], "scores": []},
        # Deceleration Rate to Avoid a Crash
        "drac": {"values": [],  "detections": [], "scores": []},
        "agent_scores": np.zeros(shape=num_agents),
        "agent_ids": [], #[(i, j) for i, j in agent_combinations],
        "agent_types": [(object_types[i], object_types[j]) for i, j in agent_combinations]
    }

    if hist_only:
        trajectories = trajectories[:, :hist_len]
        valid_masks = valid_masks[:, :hist_len]
        timesteps = hist_len
        interp_trajectories = interp_trajectories[:, :hist_len]
        interp_velocities = interp_velocities[:, :hist_len]

    # conflict points ['static', 'dynamic', 'lane_intersections'] -> (N, 2)
    conflict_points_cat = [] 
    for k, v in conflict_points.items():
        if len(v) > 0:
            conflict_points_cat.append(v[:, :2])
    conflict_points = np.concatenate(conflict_points_cat)

    dists_to_conflict_points = compute_dists_to_conflict_points(
        conflict_points, interp_trajectories, use_interp, use_extrap)
    
    for n, (i, j) in enumerate(agent_combinations):
        state['agent_ids'].append([i, j])
        for metric in supported_metrics:
            state[metric]['values'].append([])
            state[metric]['detections'].append([])
            state[metric]['scores'].append([])

        mask = None
        time = np.arange(0, timesteps / hz, 1 / hz)

        pos_i = interp_trajectories[i, :, IPOS_XY_IDX].T
        pos_j = interp_trajectories[j, :, IPOS_XY_IDX].T
        
        if np.all(np.isnan(pos_i)) or np.all(np.isnan(pos_j)):
            state['status'][n] = Status.NOK_NAN
            continue

        object_type_i, object_type_j = object_types[i], object_types[j]
        vel_i, vel_j = interp_velocities[i], interp_velocities[j]
        
        heading_i = np.rad2deg(np.arctan2(vel_i[:, 1], vel_i[:, 0]))
        heading_j = np.rad2deg(np.arctan2(vel_j[:, 1], vel_j[:, 0]))

        # Skip pedestrian-pedestrian interactions 
        if object_type_i == 'TYPE_PEDESTRIAN' and object_type_j == 'TYPE_PEDESTRIAN':
            state['status'][n] = Status.NOK_AGENT_TYPE
            continue
        
        dist_cp_i, dist_cp_j = dists_to_conflict_points[i], dists_to_conflict_points[j]

        # Get agent pair information 
        if use_interp and not use_extrap:
            mask_i = interp_trajectories[i, :, IVALID_IDX].T.astype(int)
            mask_j = interp_trajectories[j, :, IVALID_IDX].T.astype(int)
            mask = np.where(mask_i & mask_j == 1)[0]
            if mask.shape[0] < 1:
                state['status'][n] = Status.NOK_MASK
                continue
            start, end = mask[0], mask[-1]+1
            time = time[start:end]
            pos_i, pos_j = pos_i[start:end], pos_j[start:end]
            vel_i, vel_j = vel_i[start:end], vel_j[start:end]
            heading_i, heading_j = heading_i[start:end], heading_j[start:end]
            dist_cp_i, dist_cp_j = dist_cp_i[:, start:end], dist_cp_j[:, start:end]  
            
        # Skip if both agents are stationary
        v_i = np.linalg.norm(vel_i, axis=-1) 
        v_j = np.linalg.norm(vel_j, axis=-1)
        is_stationary_i = v_i.mean() <= speed_threshold
        is_stationary_j = v_j.mean() <= speed_threshold
        if is_stationary_i and is_stationary_j:
            state['status'][n] = Status.NOK_STATIONARY
            continue

        # Skip if agents are not within a distance threhold from each other
        d_ij = np.linalg.norm(pos_i - pos_j, axis=1)
        if not np.any(d_ij < agent_agent_dist_threshold):
            state['status'][n] = Status.NOK_CDIST
            continue

        # Skip if the distance between the agents is increasing over time
        if is_increasing(d_ij, time): 
            state['status'][n] = Status.NOK_SEPARATION
            continue
        
        len_i, len_j = agent_dims[i, 0], agent_dims[j, 0]

        in_cp_i = (dist_cp_i < agent_scene_conflict_dist_thresh).sum(axis=1)
        in_cp_j = (dist_cp_j < agent_scene_conflict_dist_thresh).sum(axis=1)

        agent_i = (pos_i, vel_i, heading_i, len_i, object_type_i, is_stationary_i, in_cp_i, dist_cp_i)
        agent_j = (pos_j, vel_j, heading_j, len_j, object_type_j, is_stationary_j, in_cp_j, dist_cp_j)

        # ------------------------------------------------------------------------------------------
        # Compute mTTCP for conflict points in the scene. I divided mTTCP into two metrics:
        #     * Scence mTTCP: Since all we have are points for regions like crosswalk, speed bump, 
        #                     traffic light and stop sights these are the regions from which I 
        #                     obtained conflict points. I also use approximate lane intersections. 
        #     * Agent mTTCP: I also consider if given two agent trajectories pass through the same
        #                    point. Once identified, I calculate the mTTCP from t=0 to t=first time
        #                    one of the agents cross that conflict point. This (I think), is aligned
        #                    with what's done in the ExiD paper (same PI as the analysis framework:
        #                     https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9827305)
        # ------------------------------------------------------------------------------------------
        scene_mttcp, scene_min_t, scene_shared_cps = compute_scene_mttcp(agent_i, agent_j, time)
        state['scene_mttcp']['values'][-1] = scene_mttcp
        scene_mttcp_detections = scene_mttcp <= time_threshold
        state['scene_mttcp']['detections'][-1] = scene_mttcp_detections
        if scene_mttcp_detections.sum() > 0:
            scene_mttcp_score = 1.0 / scene_mttcp[state['scene_mttcp']['detections'][-1]]
            state['scene_mttcp']['scores'][-1] = scene_mttcp_score
            state['agent_scores'][i] += scene_mttcp_score.max()
            state['agent_scores'][j] += scene_mttcp_score.max()

        agent_mttcp, agent_min_t, agent_shared_cps = compute_agent_mttcp(
            agent_i, agent_j, time, agent_agent_conflict_dist_thresh)
        state['agent_mttcp']['values'][-1] = agent_mttcp
        agent_mttcp_detections = agent_mttcp <= time_threshold
        state['agent_mttcp']['detections'][-1] = agent_mttcp_detections
        if agent_mttcp_detections.sum() > 0:
            agent_mttcp_score = 1.0 / agent_mttcp[state['agent_mttcp']['detections'][-1]]
            state['agent_mttcp']['scores'][-1] = agent_mttcp_score
            state['agent_scores'][i] += agent_mttcp_score.max()
            state['agent_scores'][j] += agent_mttcp_score.max()

        # ------------------------------------------------------------------------------------------
        # Compute THW, TTC, DRAC, which currently, all depend on whether the agents share a lane
        # at some point in their trajectories. We don't need to make this assumption but I think it
        # simplifies handling cases. Also, these metrics are usually considered for agents sharing 
        # a lane and for simple cases where the leader/follower are clearly and easily identified. 
        # ------------------------------------------------------------------------------------------

        # Determine the leading agent. NOTE: still not sure how to deal with this. First, check if 
        # the agents' headings are within a valid threshold. Then use the valid headings to determine
        # who's leading at each step based on their positions.  
        heading_diff = np.abs((heading_j - heading_i + 540) % 360 - 180)
        valid_heading = np.where(heading_diff < heading_threshold)[0]
        if valid_heading.shape[0] < 2:
            state['status'][n] = Status.NOK_HEADING
            continue
        
        leading_agent = compute_leading_agent(agent_i, agent_j, valid_heading)

        # For the rest of the metrics, the agents must be sharing a lane along their sequences. 
        if not is_sharing_lane(closest_lanes[i], closest_lanes[j]):
            state['status'][n] = Status.NOK_LANE
            continue        
        
        thw, ttc, drac = compute_interaction_state(
            agent_i, agent_j, leading_agent, time, d_ij, valid_heading)
        
        state['status'][n] = Status.OK
        state['thw']['values'][-1] = thw
        thw_detections = thw <= time_threshold
        state['thw']['detections'][-1] = thw_detections
        if thw_detections.sum() > 0:
            thw_score = 1.0 / thw[state['thw']['detections'][-1]]
            state['thw']['scores'][-1] = thw_score
            state['agent_scores'][i] += thw_score.max()
            state['agent_scores'][j] += thw_score.max()

        state['ttc']['values'][-1] = ttc
        ttc_detections = ttc <= time_threshold
        state['ttc']['detections'][-1] = ttc_detections
        if ttc_detections.sum() > 0:
            ttc_score = 1.0 / ttc[state['ttc']['detections'][-1]]
            state['ttc']['scores'][-1] = ttc_score
            state['agent_scores'][i] += ttc_score.max()
            state['agent_scores'][j] += ttc_score.max()

        state['drac']['values'][-1] = drac
        drac_detections = drac >= deceleration_threshold
        state['drac']['detections'][-1] = drac_detections
        if drac_detections.sum() > 0:
            drac_score = drac[state['drac']['detections'][-1]]
            state['drac']['scores'][-1] = drac_score
            state['agent_scores'][i] += drac_score.max()
            state['agent_scores'][j] += drac_score.max()

    if plot:
        num_windows = 1
        point_size = 1
        alpha = 0.5
        fig, ax = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        static_map_infos = scenario['map_infos']
        dynamic_map_infos = scenario['dynamic_map_infos']

        plot_static_map_infos(static_map_infos, ax, num_windows=num_windows)
        plot_dynamic_map_infos(dynamic_map_infos, ax, num_windows=num_windows)
        
        scores = state['agent_scores']
        if not np.allclose(scores, 0.0):
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = np.clip(scores, 0.2, 1.0)

        for n in range(num_agents):
            # Uses interpolated and extrapolated regions 
            if np.any(np.isnan(interp_trajectories[n, :, IPOS_XY_IDX])):
                continue

            pos = interp_trajectories[n, :, IPOS_XY_IDX].T
            if use_interp and not use_extrap:
                # Uses interpolated regions only
                mask = np.where(interp_trajectories[n, :, -1] == 1)[0]
                if mask.shape[0] == 0:
                    continue
                pos = interp_trajectories[n, mask[0]:(mask[-1]+1), IPOS_XY_IDX].T

            ax.plot(pos[:, 0], pos[:, 1], color='blue', alpha=scores[n])

        ax.set_title(f'Scene Interaction Score: {round(scores.sum(), 3)}')

        ax.set_xticks([])
        ax.set_yticks([])
            
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(
            f"{tag}_interaction_score_{round(state['agent_scores'].sum(), 3)}.png", 
            dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
    return state
