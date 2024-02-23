import pickle
import os

from legged_gym.filters.filter_helpers import wrap_to_pi, plot_state_traj, plot_state_cov_mat

if __name__ == '__main__':

	path = os.path.dirname(os.path.abspath(__file__))
	# bla = 'kf_data_03_03_2023-17-13-18_200.pickle'
	# bla = 'kf_data_03_03_2023-17-13-14_100.pickle'
	# bla = 'kf_data_06_03_2023-09-57-44_100.pickle'
	# bla = 'kf_data_06_03_2023-10-28-17_400.pickle' # +1.5
	# bla = 'kf_data_06_03_2023-10-20-42_300.pickle'  # -1.5

	# bla = 'kf_data_09_03_2023-11-22-56_200.pickle' # 600 iter policy
	# bla = 'kf_data_09_03_2023-11-22-31_100.pickle'  # 200 iter policy

	# bla = 'kf_data_09_03_2023-12-52-13_300.pickle'

	# bla = 'kf_data_21_04_2023-09-57-37_200.pickle' # xvel
	# bla = 'kf_data_21_04_2023-09-59-55_200.pickle' # yvel
	# bla = 'kf_data_21_04_2023-10-02-07_200.pickle'# angvel


	bla = 'kf_data_24_04_2023-12-09-35_50.pickle'
	# bla = 'kf_data_24_04_2023-12-09-42_200.pickle'

	# bla = 'kf_data_24_04_2023-12-11-55_200.pickle'
	# bla = 'kf_data_24_04_2023-12-11-50_100.pickle'

	filename = path + '/data/' + bla 
	with open(filename, "rb") as f:
		data_dict = pickle.load(f)

	kf = data_dict["kf"]
	real_traj = data_dict["real_traj"]#[:, :10, :]
	est_traj = data_dict["est_traj"]#[:, :10, :]
	z_traj = data_dict["z_traj"]#[:, :10, :]
	P_traj = data_dict["P_traj"]#[:, :10, :, :]
	pred_traj = None

	# print(real_traj[:, -1])

	print("shape of real_traj: ", real_traj.shape)
	print("shape of est_traj: ", est_traj.shape)
	print("shape of z_traj: ", z_traj.shape)
	print("shape of P_traj: ", P_traj.shape)

	plot_state_traj(real_traj, z_traj, pred_traj, est_traj, show_fig=True)
	plot_state_cov_mat(P_traj, est_traj, show_fig=True)

