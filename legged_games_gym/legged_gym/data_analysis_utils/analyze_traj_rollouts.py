import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datetime import datetime
import os

import pickle
from pretty_plt_utils import plot_joint_traj

if __name__ == '__main__':

    # ================================= #
    # TEACHER: Policy design ablations
    # ================================= #
    folder = "teacher_policy_ablations"
    name = "Vis_Agent_simple_weaving_Robot_reaction_Policyreactive_policy" 			# REACTIVE Robot w/ HEURISTIC (simple weaving) opponent
    # name = "Vis_Agent_simple_weaving_Robot_prediction_phase1_Policylstm_estimation_policy" 			# ESTIMATION Robot w/ HEURISTIC (simple weaving) opponent
    # name = "Vis_Agent_simple_weaving_Robot_prediction_phase2_Policyp2_simpleWeave_randEverything_lstm" 			# PREDICTION Robot w/ HEURISTIC (simple weaving) opponent

    fname = "reaction"
    # fname = "estimation"
    # fname = "prediction"

    # read files
    path = os.path.dirname(os.path.abspath(__file__))
    filename = path + '/data/' + name + ".pickle"
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)

    # extract the data
    agent_state_data = data_dict["agent_state_data"]
    robot_state_data = data_dict["robot_state_data"]
    rel_pos_data = data_dict["rel_pos_data"]
    rel_state_robot_frame_data = data_dict["rel_state_robot_frame_data"]
    cur_episode_length = data_dict["cur_episode_length"]
    capture_buf = data_dict["capture_buf"]

    print("Plotting raw trajectory data...")
    env_idx = 0
    plot_joint_traj(robot_state_data[env_idx, :, :],
                    agent_state_data[env_idx, :, :],
                    show_fig=False, filename=fname)



