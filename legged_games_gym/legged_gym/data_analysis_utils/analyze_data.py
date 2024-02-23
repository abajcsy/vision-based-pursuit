import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datetime import datetime
import os

import pickle
from pretty_plt_utils import plot_traj_data, plot_capture_time_heatmap, plot_capture_time_distribution, plot_theoretic_capture_time

if __name__ == '__main__':

	# ================================= # 
	# TEACHER: Policy design ablations
	# ================================= # 
	# folder = "teacher_policy_ablations"
	# name = "Agent_simple_weaving_Robot_reaction_Policyreactive_policy" 			# REACTIVE Robot w/ HEURISTIC (simple weaving) opponent
	# name = "" 			# ESTIMATION Robot w/ HEURISTIC (simple weaving) opponent
	# name = "Agent_simple_weaving_Robot_prediction_phase2_Policyp2_simpleWeave_randEverything_lstm" 			# PREDICTION Robot w/ HEURISTIC (simple weaving) opponent

	# ================================= # 
	# TEACHER: Fully observable policies
	# ================================= # 
	# folder = "full_vs_partial_obs"
	# name = "Agent_simple_weaving_Robot_prediction_phase1_Policyphase_1_policy_v3" 	# HEURISTIC (simple weaving)
	# name = "Agent_complex_weaving_Robot_prediction_phase2_Policyp2_complexWeave_lstm" 	# HEURISTIC (complex weaving)
	# name = "Agent_complex_weaving_Robot_prediction_phase2_Policyp2_marl_lstm"   			# MARL
	# name = "v2_Agent_learned_traj_Robot_prediction_phase2_Policyp2_marl_lstm"
	# name = "Agent_game_theory_Robot_game_theory" 										# GAME THEORY

	# ====================== # 
	# STUDENT: PO policies
	# ====================== # 
	# folder = "full_vs_partial_obs"
	# name = "Agent_simple_weaving_Robot_po_prediction_phase2_Policypo_p2_complexWeave_lstm"		  	# HEURISTIC (simple wevaing)
	# name = "Agent_complex_weaving_Robot_po_prediction_phase2_Policypo_p2_complexWeave_lstm"		  	# HEURISTIC (complex wevaing)
	# name = "Agent_learned_traj_Robot_po_prediction_phase2_Policypo_p2_marl_lstm"    					# MARL
	# name = "v2_Agent_learned_traj_Robot_po_prediction_phase2_Policypo_p2_marl_lstm"
	# name = "Agent_game_theory_Robot_po_prediction_phase2_PolicyMay29_20-32-51_po_game_theory_teacher" # GAME THEORY

	# ============================== # 
	# OOD Opponent w/ PO Ego policies
	# ============================== # 
	folder = "ood_opponent"
	# name = "Agent_complex_weaving_Robot_po_prediction_phase2_PolicyMay29_20-32-51_po_game_theory_teacher"			# GAME ego w/ HEURISTIC opponent
	# name = "Agent_simple_weaving_Robot_po_prediction_phase2_PolicyMay29_20-32-51_po_game_theory_teacher"
	# name = "Agent_complex_weaving_Robot_po_prediction_phase2_Policypo_p2_marl_lstm"  						# MARL ego w/ HEURISTIC opponent
	name = "Agent_game_theory_Robot_po_prediction_phase2_Policypo_p2_marl_lstm" 		# MARL ego w/ GAME opponent
	# name = ""  						# MARL ego w/ HEURISTIC (complex weave) opponent
	# name = "Agent_game_theory_Robot_po_prediction_phase2_Policypo_p2_complexWeave_lstm" # HEURISTIC ego w/ GAME opponent
        # name = "Agent_learned_traj_Robot_po_prediction_phase2_Policypo_p2_complexWeave_lstm"   		# HEURISTIC ego w/ MARL opponent
	# name = "Agent_learned_traj_Robot_po_prediction_phase2_PolicyMay29_20-32-51_po_game_theory_teacher"	# GAME ego w/ MARL opponent

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

	# if running the game-theory policies, then we can compute theoretic capture time.
	if "theoretic_capture_time" in data_dict:
		theoretic_capture_time = data_dict["theoretic_capture_time"]
		print("Plotting *theoretic* capture time heatmap...")
		plot_theoretic_capture_time(rel_state_robot_frame_data, 
									theoretic_capture_time, 
									show_fig=False, 
									filename=name, 
									folder=folder)

	# plot actual capture time
	print("Plotting capture time heatmap...")
	plot_capture_time_heatmap(rel_state_robot_frame_data, 
								cur_episode_length, 
								show_fig=False, 
								filename=name, 
								folder=folder)

	# print("Plotting capture time distribution...")
	# plot_capture_time_distribution(cur_episode_length, show_fig=False, filename=name)

	# print("Plotting raw trajectory data...")
	# plot_traj_data(rel_pos_data, robot_data=robot_state_data, show_fig=False, filename=name)



