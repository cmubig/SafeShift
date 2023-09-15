import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from utils.common import (
    mph_to_ms, IPOS_XY_IDX, ILANE_IDX, IVALID_IDX, AGENT_TYPE_MAP
)
from utils.visualization import plot_static_map_infos, plot_dynamic_map_infos

np.set_printoptions(suppress=True, precision=4)

# --------------------------------------------------------------------------------------------------
# Feature Wrapper
# --------------------------------------------------------------------------------------------------
def compute_individual_features(
    scenario: dict, conflict_points: np.array, load_type: str, interp_trajectories: np.array, 
    interp_velocities = None, cluster_anomaly: dict = None, plot: bool = False, tag: str = 'temp', timesteps = 91, hz = 10
):
    """ Measures the individual state of all agents. """
    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    trajectories = track_infos['trajs'][:, :, :-1]
    object_types = track_infos['object_type']

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    dynamic_map_infos = scenario['dynamic_map_infos']
    static_map_infos = scenario['map_infos']
    lane_info = static_map_infos['lane']

    num_agents, _, _ = trajectories.shape
    state = {
        "speed": {"values": [], "lane_limit_diff": []},
        "acc": {"ax": [], "ay": []},
        "jerk": {"jx": [], "jy": []},
        "yaw_rate": [],
        "waiting_period": {"intervals": [], "min_dists_conflict_points": [], 'values':[]},
        "valid_agents": [],
        "agent_types": [], #[AGENT_TYPE_MAP[object_type] for object_type in object_types],
        "in_lane": [],
        "n_lanes": [],
        # TODO: technically need to move this to just be appended in the normal iteration, rather than copying the whole thing.
        # Re-assess this after all the other experiments are done, for consistency
        "traj_anomaly": cluster_anomaly['singles']
    }
    
    # conflict points ['static', 'dynamic', 'lane_intersections'] -> (N, 2)
    cp_list = [] 
    cpts = None
    for k, v in conflict_points.items():
        if len(v) > 0:
            cp_list.append(v[:, :2])
    if len(cp_list) > 0:
        cpts = np.concatenate(cp_list)

    # TODO: add load_type == 'ho'

    num_agents, _, _ = trajectories.shape
    for n in range(num_agents):
        lane_idx = interp_trajectories[n, :, ILANE_IDX].T
        pos = interp_trajectories[n, :, IPOS_XY_IDX].T

        # nan -> no sequence found; check if there are cases where 
        # inf -> partial sequence (less likely for you to rejoin the lane later)
        # if np.any(np.isnan(pos)) or not np.any(np.isfinite(lane_idx)): 
        if np.any(np.isnan(pos)): 
            continue
        
        # It's okay to not have a lane assignment I think, or maybe it would simplify things a bunch if we only looked at cases 
        # where people are actually traveling in lanes? But that would throw out pedestrians, so idk
        # if not np.any(np.isfinite(lane_idx)):
        #     continue

        mask = None
        time = np.arange(0, timesteps / hz, 1 / hz)

        # If using extrapolated futures, we're not masking anything. 
        agent_type = AGENT_TYPE_MAP[object_types[n]]
        vel = interp_velocities[n]
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        
        # If using GT, mask extrapolations. 
        if load_type == 'gt':
            mask = np.where(interp_trajectories[n, :, -1] == 1)[0]
            if mask.shape[0] == 0:
                continue
            start, end = mask[0], mask[-1]+1
            
            pos = interp_trajectories[n, start:end, IPOS_XY_IDX].T
            vel = interp_velocities[n, start:end]
            heading = np.arctan2(pos[:, 1], pos[:, 0])
            lane_idx = lane_idx[start:end].astype(int)
            
            time = time[start:end]

        if pos.shape[0] < 2:
            continue

        # speed_limits = np.array(
        #     [mph_to_ms(lane_info[lane_idx[i][0]]['speed_limit_mph']) for i in range(lane_idx.shape[0])])
        speed_limits = np.zeros(pos.shape[0])
        in_lane = np.zeros(pos.shape[0]).astype(bool)
        for i in range(lane_idx.shape[0]):
            speed_limits[i] = speed_limits[-1]
            if lane_idx[i][0] > 0 and lane_idx[i][0] < len(lane_info):
                speed_limits[i] = mph_to_ms(lane_info[int(lane_idx[i][0])]['speed_limit_mph'])
                in_lane[i] = True
            else:
                in_lane[i] = False
        
        # Add agent state
        speed, speed_limit_diff, acc_x, acc_y, jerk_x, jerk_y, yaw_rate = compute_individual_state(
            vel, time, heading, speed_limits, hz)
        state['speed']['values'].append(speed)
        state['speed']['lane_limit_diff'].append(speed_limit_diff)
        state['acc']['ax'].append(acc_x)
        state['acc']['ay'].append(acc_y)
        state['jerk']['jx'].append(jerk_x)
        state['jerk']['jy'].append(jerk_y)
        state['yaw_rate'].append(yaw_rate)
        state['in_lane'].append(in_lane)
        state['n_lanes'].append(len(np.unique(lane_idx[np.isfinite(lane_idx)])))
    
        wp_intervals, wp_min_dist_cp, wp_values = compute_waiting_period(pos, time, cpts)
        state['waiting_period']['intervals'].append(wp_intervals)
        state['waiting_period']['min_dists_conflict_points'].append(wp_min_dist_cp)
        state['waiting_period']['values'].append(wp_values)
        
        state['valid_agents'].append(n)
        state['agent_types'].append(agent_type)

    if plot:
        num_windows = 6
        fig, ax = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))
        # Map infos:
        #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
        static_map_infos = scenario['map_infos']
        dynamic_map_infos = scenario['dynamic_map_infos']

        plot_static_map_infos(static_map_infos, ax)
        plot_dynamic_map_infos(dynamic_map_infos, ax)

        valid_agents = state['valid_agents']

        max_speed = 1.0
        for speed in state['speed']['values']:
            max_speed = max(speed.max(), max_speed)

        # TODO: fix
        # max_wp = state['waiting_period']['intervals']
        # if np.any(max_wp > 0):
        #     max_wp /= max_wp.max()

        valid_agent = 0
        for n in range(num_agents):
            # Uses interpolated and extrapolated regions 
            if np.all(np.isnan(interp_trajectories[n, :, IPOS_XY_IDX])):
                continue
            
            pos = interp_trajectories[n, :, IPOS_XY_IDX].T
            if n not in valid_agents:
                for a in ax.reshape(-1):
                    a.plot(pos[:, 0], pos[:, 1], color='red', alpha=0.5)
                continue

            if load_type == 'gt':
                # Uses interpolated regions only
                mask = np.where(interp_trajectories[n, :, -1] == 1)[0]
                if mask.shape[0] == 0:
                    continue
                pos = interp_trajectories[n, mask[0]:(mask[-1]+1), IPOS_XY_IDX].T

            speed = state['speed']['values'][valid_agent]
            speed /= max_speed
            size = 0.5
            ax[0].scatter(pos[:, 0], pos[:, 1], s=size, c=cm.winter(speed))
            ax[0].set_title('Speed')

            acc_x = state['acc']['ax'][valid_agent]
            if acc_x.max() != 0:
                acc_x = (acc_x - acc_x.min()) / (acc_x.max() - acc_x.min())
            ax[1].scatter(pos[:, 0], pos[:, 1], s=size, c=cm.winter(acc_x))
            ax[1].set_title('Acc x')

            acc_y = state['acc']['ay'][valid_agent]
            if acc_y.max() != 0:
                acc_y = (acc_y - acc_y.min()) / (acc_y.max() - acc_y.min())
            ax[2].scatter(pos[:, 0], pos[:, 1], s=size, c=cm.winter(acc_y))
            ax[2].set_title('Acc y')
            
            jerk_x = state['jerk']['jx'][valid_agent]
            if jerk_x.max() != 0:
                jerk_x = (jerk_x - jerk_x.min()) / (jerk_x.max() - jerk_x.min())
            ax[3].scatter(pos[:, 0], pos[:, 1], s=size, c=cm.winter(jerk_x))
            ax[3].set_title('Jerk x')

            jerk_y = state['jerk']['jy'][valid_agent]
            if jerk_y.max() != 0:
                jerk_y = (jerk_y - jerk_y.min()) / (jerk_y.max() - jerk_y.min())
            ax[4].scatter(pos[:, 0], pos[:, 1], s=size, c=cm.winter(jerk_x))
            ax[4].set_title('Jerk y')

            yaw_rate = state['yaw_rate'][valid_agent]
            if yaw_rate.max() != 0:
                yaw_rate = (yaw_rate - yaw_rate.min()) / (yaw_rate.max() - yaw_rate.min())
            ax[5].scatter(pos[:, 0], pos[:, 1], s=size, c=cm.winter(yaw_rate))
            ax[5].set_title('Yaw Rate')

            # ax[4].plot(pos[:, 0], pos[:, 1], c=cm.winter(max_wp[valid_agent]))
            # ax[4].set_title('Waiting Period')

            valid_agent += 1

        for a in ax.reshape(-1):
            a.set_xticks([])
            a.set_yticks([])

        plt.subplots_adjust(wspace=0.05)
        plt.suptitle(f'Individual Vehicle state')
        plt.savefig(f"{tag}_individual_{load_type}.png", dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()

    return state
# --------------------------------------------------------------------------------------------------
# Scores wrapper. Initial attempt at implementing the scoring mechanism of the analysis framework
# paper. 
# --------------------------------------------------------------------------------------------------
def get_acc_lon_detections(acc_lon, speed, speed_ranges):
    raise NotImplementedError
    m50, m100 = km_to_miles(50.0), km_to_miles(100.0)
    m2, m4 = meters_to_miles(2.0), meters_to_miles(4.0)

    leq50 = speed_ranges['leq50']
    gt50_lt100 = speed_ranges['gt50_leq100']
    gt100 = speed_ranges['gt100']

    acc_lon_det = np.zeros_like(acc_lon)
    acc_lon_det[leq50] = np.abs(acc_lon[leq50]) > m4
    acc_lon_det[gt50_lt100] = np.abs(acc_lon[gt50_lt100]) > np.abs(m4 - m2 * (speed[gt50_lt100] - m50) / m100)
    acc_lon_det[gt100] = np.abs(acc_lon[gt100]) > m2
    return acc_lon_det

def get_acc_lat_detections(acc_lat, speed, speed_ranges):
    raise NotImplementedError
    m40, m50 = km_to_miles(40.0), km_to_miles(50.0)
    m2_5, m3, m4 = meters_to_miles(2.5), meters_to_miles(3.0), meters_to_miles(4.0)
    m4_5, m7 = meters_to_miles(4.5), meters_to_miles(7.0)

    leq40 = speed_ranges['leq40']
    gt40_lt50 = speed_ranges['gt40_leq50']
    gt50_lt100 = speed_ranges['gt50_leq100']
    gt100 = speed_ranges['gt100']

    abs_acc_lat = np.abs(acc_lat)

    acc_lat_det = np.zeros_like(acc_lat)
    acc_lat_det[leq40] = abs_acc_lat[leq40] > m2_5 + m4_5 * (speed[leq40] / m40)
    acc_lat_det[gt40_lt50] = abs_acc_lat[gt40_lt50] > m7
    acc_lat_det[gt50_lt100] = abs_acc_lat[gt50_lt100] > m7 - m4 * ((speed[gt50_lt100] - m50) / m50)
    acc_lat_det[gt100] = abs_acc_lat[gt100] > m3
    return acc_lat_det

def get_yaw_rate_detections(yaw_rate, speed_ranges):
    raise NotImplementedError
    leq50 = speed_ranges['leq50']
    gt50 = speed_ranges['gt50']

    abs_yaw_rate = np.abs(yaw_rate)
    yaw_rate_det = np.zeros_like(yaw_rate)
    yaw_rate_det[leq50] = abs_yaw_rate[leq50] > 50.0/180.0 # deg /s
    yaw_rate_det[gt50] = abs_yaw_rate[gt50] > 15.0/180.0 # deg /s
    return yaw_rate_det

def get_sideslip_detections(sideslip, sideslip_threshold = 0.1745):
    raise NotImplementedError
    abs_sideslip = np.abs(sideslip)
    sideslip_det = np.zeros_like(sideslip)
    sideslip_det = abs_sideslip > sideslip_threshold # 10 deg
    return sideslip_det

# --------------------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------------------
def compute_individual_state(velocity, time, heading, speed_limits, hz):
    """ Computes the individual agent state:
            - speed
            - yaw rate
            - acceleration 
            - jerk
    """
    speed = np.linalg.norm(velocity, axis=-1)
    speed_limit_diff = speed - speed_limits

    yaw_rate = np.gradient(heading, time)
    # See: https://stackoverflow.com/a/7869457/10101616
    # yaw_rate = np.zeros_like(heading)
    # yaw_rate[1:] = heading[1:] - heading[:-1]
    # yaw_rate[0] = yaw_rate[1]
    yaw_rate = ((yaw_rate + np.pi) % (2*np.pi)) - np.pi
    # yaw_rate *= hz
    
    acc_x = np.gradient(velocity[:, 0], time)
    acc_y = np.gradient(velocity[:, 1], time) 

    jerk_x = np.gradient(acc_x, time)
    jerk_y = np.gradient(acc_y, time)

    # Lateral acceleration: https://www.sciencedirect.com/topics/engineering/lateral-acceleration
    # sideslip = -np.arctan2(vel[:, 1], np.abs(vel[:, 0]))
    # dside_slip = sideslip[1:] - sideslip[:-1]
    # side_slip_rate = dside_slip / dt
    # acc_lat = dv * yaw_rate + side_slip_rate

    return speed, speed_limit_diff, acc_x, acc_y, jerk_x, jerk_y, yaw_rate

def compute_waiting_period(pos, time, conflict_points = None, motion_threshold = 0.5):
    wp_intervals, wp_values, wp_dist_conflict_point = \
        np.zeros(shape=1), np.zeros(shape=1), np.inf * np.ones(shape=1)

    dt = time[1:] - time[:-1]
    dp = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    #is_waiting = np.where(dp < motion_threshold)[0]
    is_waiting = dp < motion_threshold
    if sum(is_waiting) > 0:
        # From https://stackoverflow.com/a/29853487/10101616
        is_waiting = np.hstack([ [False], is_waiting, [False] ])  # padding
        is_waiting = np.diff(is_waiting.astype(int))
        starts = np.where(is_waiting == 1)[0]
        ends = np.where(is_waiting == -1)[0]

        wp_intervals = np.array([dt[start:end].sum() for start, end in zip(starts, ends)])
        
        if not conflict_points is None and len(starts) > 0:
            wp_dist_conflict_point = np.linalg.norm(conflict_points[:, None] - pos[starts], axis=-1).min(axis=0)

        # TODO: Think of some other way of combining them
        wp_values = wp_intervals * (1.0 / wp_dist_conflict_point)

    return wp_intervals, wp_dist_conflict_point, wp_values

def compute_speed(velocity, mask = None):
    speed = np.linalg.norm(velocity, axis=-1)
    if np.isnan(speed).any():
        raise ValueError(f"Nan value in agent speed: {speed} velocity: {velocity}")
    if mask is None:
        return speed.mean(), speed.min(), speed.max(), speed.argmin(), speed.argmax()
    return speed.mean(), speed.min(), speed.max(), mask[speed.argmin()], mask[speed.argmax()]

def compute_sideslip(velocity, mask = None):
    sideslip = -np.arctan2(velocity[:, 1], np.abs(velocity[:, 0]))
    # using this formula: https://en.wikipedia.org/wiki/Slip_angle
    if np.isnan(sideslip).any():
        raise ValueError(f"Nan value in agent sideslip: {sideslip} velocity: {velocity}")
    if mask is None:
        return sideslip.mean(), sideslip.min(), sideslip.max(), sideslip.argmin(), sideslip.argmax()
    return sideslip.mean(), sideslip.min(), sideslip.max(), mask[sideslip.argmin()], mask[sideslip.argmax()]

def compute_yaw_rate(heading, time, mask = None):
    if not mask is None:   
        time = time[mask]
    dt = time[1:] - time[:-1]
    dh = heading[1:] - heading[:-1]
    yaw_rate = dh / dt 
    if np.isnan(yaw_rate).any():
        raise ValueError(f"Nan value in agent yaw rate: {yaw_rate} heading: {heading}heading")
    
    if mask is None:
        return yaw_rate.mean(), yaw_rate.min(), yaw_rate.max(), yaw_rate.argmin(), yaw_rate.argmax()
    return yaw_rate.mean(), yaw_rate.min(), yaw_rate.max(), mask[yaw_rate.argmin()], mask[yaw_rate.argmax()]

def compute_individual_scores(
    scenario: dict, conflict_points: np.array, use_interp: bool = False, use_extrap: bool = False, 
    hist_only: bool = False, interp_trajectories = None, interp_velocities = None, plot: bool = False, 
    tag: str = 'temp', timesteps = 91, hz = 10, hist_len = 11, occgrid_size: int = 10
):
    raise NotImplementedError
    """ Computes trajectories individual scores. """
    state = {}

    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    trajectories = track_infos['trajs'][:, :, :-1]
    valid_masks = track_infos['trajs'][:, :, -1] > 0
    object_types = track_infos['object_type']
    
    # Interp trajectory data
    #   x, y (cartesian), s,d,z (frenet), lane_idx, valid

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    breakpoint()
    dynamic_map_infos = scenario['dynamic_map_infos']
    static_map_infos = scenario['map_infos']
    lane_info = static_map_infos['lane']
    
    polylines = static_map_infos['all_polylines'][:, :2]
    x_max, y_max = polylines.max(axis=0)
    x_min, y_min = polylines.min(axis=0)

    dx, dy = (x_max - x_min) / occgrid_size, (y_max - y_min) / occgrid_size
    X, Y = np.arange(x_min, x_max, dx), np.arange(y_min, y_max, dy)

    num_features = 5
    num_count_types = 2 # det count, all counts
    occgrid = np.zeros(shape=(occgrid_size, occgrid_size, num_features, num_count_types)) 

    if hist_only:
        trajectories = trajectories[:, :hist_len]
        valid_masks = valid_masks[:, :hist_len]
        timesteps = hist_len
        if use_interp:
            interp_trajectories = interp_trajectories[:, :hist_len]
            interp_velocities = interp_velocities[:, :hist_len]

    num_agents, _, _ = trajectories.shape

    state = {
        "agent_id": [],
        "speed": {"values": [], "detections": []},
        "acc_lon": {"values": [], "detections": []},
        "acc_lat": {"values": [], "detections": []},
        "yaw_rate": {"values": [], "detections": []},
        "sideslip": {"values": [], "detections": []},
        "agent_types": [], # [AGENT_TYPE_MAP[object_type] for object_type in object_types],
        'x_idxs': [], 
        'y_idxs': [],
    }

    for n in range(num_agents):
        mask = None
        time = np.arange(0, timesteps / hz, 1 / hz)

        pos = interp_trajectories[n, :, IPOS_XY_IDX].T
        lane_idx = interp_trajectories[n, :, ILANE_IDX].T
        valid = interp_trajectories[n, :, IVALID_IDX].T
        
        if np.any(np.isnan(pos)) or np.all(np.isposinf(lane_idx)):
            continue
    
        vel = interp_velocities[n]
        heading = np.arctan2(pos[:, 1], pos[:, 0])
        
        if use_interp and not use_extrap:
            # Uses interpolated regions only
            mask = np.where(valid == 1)[0]
            if mask.shape[0] < 3: # Need at least two valid pairs of points 
                continue

            start, end = mask[0], mask[-1]+1
            pos = pos[start:end]

            # TODO: fix out of range error here
            lane_idx = lane_idx[start:end].astype(int)
            raise ValueError
            speed_limits = np.array(
                [lane_info[lane_idx[i][0]]['speed_limit_mph'] for i in range(lane_idx.shape[0])])
            vel = vel[start:end]
            heading = heading[start:end]
            time = time[start:end]
        
        # ------------------------------------------------------------------------------------------
        # Compute measurements 
        state['agent_id'].append(n)
        state['agent_types'].append(AGENT_TYPE_MAP[object_types[n]])
        speed, acc_lon, acc_lat, yaw_rate, sideslip = compute_individual_state(vel, time, heading)

        # ------------------------------------------------------------------------------------------
        # Compute detections
        # NOTE: using thresholds and limits from: https://arxiv.org/pdf/2202.07438.pdf (but in mph)
        speed_detections = speed > speed_limits
        state['speed']['values'].append(speed)
        state['speed']['detections'].append(speed_detections)

        m40, m50, m100 = km_to_miles(40.0), km_to_miles(50.0), km_to_miles(100.0)
        speed = speed[1:]
        speed_ranges = {
            'leq40': np.where(speed <= m40)[0],
            'leq50': np.where(speed <= m50)[0],
            'leq100': np.where(speed <= m100)[0],
            'gt40_leq50': np.where(np.logical_and(speed > m40, speed <= m50))[0],
            'gt50_leq100': np.where(np.logical_and(speed > m50, speed <= m100))[0],
            'gt50': np.where(speed > m50)[0],
            'gt100': np.where(speed > m100)[0]
        }

        # Longitudinal acceleration 
        state['acc_lon']['values'].append(acc_lon)
        state['acc_lon']['detections'].append(get_acc_lon_detections(acc_lon, speed, speed_ranges))
        
        # Lateral acceleration
        state['acc_lat']['values'].append(acc_lat)
        state['acc_lat']['detections'].append(get_acc_lat_detections(acc_lat, speed, speed_ranges))

        # Yaw Rate
        state['yaw_rate']['values'].append(yaw_rate)
        state['yaw_rate']['detections'].append(get_yaw_rate_detections(yaw_rate, speed_ranges))
        
        # Sideslip 
        state['sideslip']['values'].append(sideslip)
        state['sideslip']['detections'].append(get_sideslip_detections(sideslip))

        # ------------------------------------------------------------------------------------------
        # Update grid for anomaly scores
        # NOTE: following same paper as above. Here, scores focus on rarity of events, where they 
        # weight the events / detections by a context: c = (1 / M) * (N / sqrt(M)), where 
        #      M is the number of detections of a type belonging to a region 
        #      N is the number of users passing that region
        # NOTE: Not sure how they handle regions. It seems that they just do it semantically?
        # For simplicity, I'm handling this as an occupancy grid. 
        # H, x_edges, y_edges = np.histogram2d(
        #     pos[:, 0], pos[:, 1], bins=occgrid_size, range=[[x_min, x_max], [y_min, y_max]])
        x_idxs = np.searchsorted(X, pos[1:, 0], side='left')
        x_idxs[np.where(x_idxs == occgrid_size)] = occgrid_size - 1
        y_idxs = np.searchsorted(Y, pos[1:, 1], side='left')
        y_idxs[np.where(y_idxs == occgrid_size)] = occgrid_size - 1

        state['x_idxs'].append(x_idxs)
        state['y_idxs'].append(y_idxs)

        for t in range(len(x_idxs)):
            x, y = x_idxs[t], y_idxs[t]

            # all counts
            occgrid[x, y, :, 1] += 1
            
            # detection counts
            occgrid[x, y, 0, 0] += int(state['speed']['detections'][-1][t])   
            occgrid[x, y, 1, 0] += int(state['acc_lon']['detections'][-1][t]) 
            occgrid[x, y, 2, 0] += int(state['acc_lat']['detections'][-1][t]) 
            occgrid[x, y, 3, 0] += int(state['yaw_rate']['detections'][-1][t]) 
            occgrid[x, y, 4, 0] += int(state['sideslip']['detections'][-1][t]) 

    # ----------------------------------------------------------------------------------------------
    # Compute anomaly scores
    total_score = 0.0
    num_valid_agents = len(state['agent_id'])
    state['anomaly_scores'] = np.zeros(shape=(num_valid_agents))
    if num_valid_agents > 0:
        for n in range(len(state['agent_id'])):
            #  c = (1 / M) * (N / sqrt(M))
            # iscore = c * * detection * value
            score = 0.0
            C = np.zeros(shape=(5))
            for t in range(len(state['x_idxs'][n])):
                x, y = state['x_idxs'][n][t], state['y_idxs'][n][t]
                M = occgrid[x, y, :, 0]
                idx = np.where(M > 0.0)
                N = occgrid[x, y, :, 1]
                C[idx] = (1 / M[idx]) * (N[idx] / np.sqrt(M[idx]))

                for i, k in enumerate(['speed', 'acc_lon', 'acc_lat', 'yaw_rate', 'sideslip']):
                    if C[i] > 0:
                        score += C[i] * state[k]['values'][n][t] * state[k]['detections'][n][t]

            state['anomaly_scores'][n] = score
            total_score += score
    
    # Plot
    if plot:
        num_windows = 1
        fig, ax = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))
        static_map_infos = scenario['map_infos']
        dynamic_map_infos = scenario['dynamic_map_infos']

        plot_static_map_infos(static_map_infos, ax, num_windows)
        plot_dynamic_map_infos(dynamic_map_infos, ax, num_windows)

        for n in range(num_agents):              
            color = 'blue' if n not in state['agent_id'] else 'red'
            if use_interp and not use_extrap:
                # Uses interpolated regions only
                mask = np.where(interp_trajectories[n, :, -1] == 1)[0]
                if mask.shape[0] == 0:
                    continue
                pos = interp_trajectories[n, mask[0]:(mask[-1]+1), IPOS_XY_IDX].T
            elif use_interp and use_extrap:
                # Uses interpolated and extrapolated regions 
                if np.any(np.isnan(interp_trajectories[n, :, IPOS_XY_IDX])):
                    continue
                pos = interp_trajectories[n, :, IPOS_XY_IDX].T
    
            ax.plot(pos[:, 0], pos[:, 1], color=color)

        ax.set_xticks(X)
        ax.set_yticks(Y)
        ax.grid(which='both', alpha=0.3)
        
        plt.subplots_adjust(wspace=0.05)
        plt.suptitle(f'Individual Vehicle state')
        plt.savefig(
            f"{tag}_individual_{use_interp}-{use_extrap}_{total_score}.png", dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()

    return state
