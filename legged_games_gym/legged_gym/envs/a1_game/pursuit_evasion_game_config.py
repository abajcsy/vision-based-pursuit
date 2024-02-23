
from legged_gym.envs.base.base_config import BaseConfig

Run_Name = '' # give a name that is informative about the run
Save_Folder_Name = 'dec_high_level_game' # Saving folder in ../../../../../logs. Keep this constant for the same family of experiments

class DecHighLevelGameCfg( BaseConfig ):
    class env:
        debug_viz = False
        robot_hl_dt = 0.2   # 1 / robot_hl_dt is the Hz

        num_envs = 4096 # 4096 for training

        num_actions_robot = 3           # pursuer (lin_vel_x, lin_vel_y, ang_vel_yaw) = 3
        num_actions_agent = 2           # evader (lin_vel, ang_vel) = 2
        num_agent_states = 3            # x = (px, py, pz)
        num_pred_steps = 8              # prediction length
        num_hist_steps = 8              # history length

        # robot policy type options:
        #       'prediction_phase1' uses privileged info about future relative state
        #       'prediction_phase2' uses history of *perfect* relative state and robot actions with privileged future predictions
        #       'po_prediction_phase2' uses a history of *estimated* state and covariant, (xhat, cov), from Kalman filter with privileged perfect-state future predictions
        robot_policy_type = 'prediction_phase1'
        
        # interaction scenario options:
        #       'nav' if doing goal-reaching w/agent avoidance (not supported at the moment)
        #       'game' pursuit-evasion interaction
        interaction_type = 'game'

        # ====== [Pursuit-Evasion Game] ====== #
        if interaction_type == 'game':
            # PREDICTION - PHASE 1 (Reinforcement Learning) 
            if robot_policy_type == 'prediction_phase1':
                # Robot
                num_robot_states = 4                                    # state dimension: x = (px, py, pz, theta). 
                num_priv_robot_states = None                            # privileged state dimension: no notion of privileged states during RL
                num_observations_robot = num_robot_states * (num_pred_steps + 1)        # observations: robot observes *future* of state trajectory (x^t:t+N)
                num_privileged_obs_robot = None                                         # privileged observations: no notion of privileged obs during RL

                # Agent
                num_observations_agent = 4                             
                num_privileged_obs_agent = None 
            # FULL OBSERVABILITY - PHASE 2 (Supervised Learning)
            elif robot_policy_type == 'prediction_phase2':
                # Robot TEACHER
                num_priv_robot_states = 4                   # privileged state dimension: x = (px, py, pz, theta).
                num_observations_priv_robot = num_priv_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps   # observations: robot observes *history* of states & actions (x^t-N:t, uR^t-N:t-1)
                num_privileged_obs_priv_robot = num_priv_robot_states * (num_pred_steps + 1)                                      # privileged observations: observing *future* of state trajectory (x^t:t+N)

                # Robot STUDENT
                num_robot_states = 4                        # state dimension: x = (px, py, pz, theta). 
                num_observations_robot = num_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps   # observations: robot observes *history* of states & actions (x^t-N:t, uR^t-N:t-1)
                num_privileged_obs_robot = num_privileged_obs_priv_robot                                      # privileged observations: observing *future* of state trajectory (x^t:t+N)

                # Agent
                num_observations_agent = 4
                num_privileged_obs_agent = None
            # PARTIAL OBSERVABILITY - PHASE 2 (Supervised Learning)
            elif robot_policy_type == 'po_prediction_phase2':
                # Robot TEACHER
                num_priv_robot_states = 4               # expert state dimension: x = (px, py, pz, theta)
                num_observations_priv_robot = num_priv_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps # expert observations: robot observes *history* of states & actions (x^t-N:t, uR^t-N:t-1)
                num_privileged_obs_priv_robot = num_priv_robot_states * (num_pred_steps + 1)                                    # privileged observations: observing *future* of state trajectory (x^t:t+N)

                # Robot STUDENT
                num_robot_states = 8                    # student state dimension: x = (px, py, pz, theta, sx, sy, sz, stheta).
                num_observations_robot = num_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps # observations: robot observes *estimate history* of states, covariances, & actions (xhat^t-N:t, Sigma^t-N:t, uR^t-N:t-1)
                num_privileged_obs_robot = num_privileged_obs_priv_robot        # privileged observations: observing *future* of state trajectory (x^t:t+N)

                # Agent
                num_observations_agent = 4
                num_privileged_obs_agent = None
        # =========== END ========== #

        env_spacing = 3.            # not used with heightfields / trimeshes
        send_timeouts = False       # send time out information to the algorithm
        send_BC_actions = True      # send optimal robot actions for the BC loss in the algorithm; (leave as is)
        episode_length_s = 20       # episode length in seconds
        capture_dist = 0.8          # if the two agents are closer than this dist, they are captured


        # simulated agent info
        # options for agent's dynamics:
        #       'dubins' (u = linvel, angvel)
        #       'integrator' (u = xvel, yvel)
        agent_dyn_type = 'dubins'

        # options for agent policy type:
        #       'learned_traj' it has a MLP predicting a trajectory (trained with MARL)
        #       'simple_weaving' it follows dubins' curves (sinusoids)
        #       'complex_weaving' it follows random linear and angular velocity combinations (random motion)
        #       'static' just stands still
        agent_policy_type = 'complex_weaving'
        agent_ang = [-3.14, 3.14]       # initial condition: [min, max] relative angle to robot
        agent_rad = [2.0, 6.0]          # initial condition: [min, max] spawn radius away from robot
        agent_facing_away = True        # initial condition: True = agent faces away from predator, False = agent is axis-aligned
        randomize_on_reset = False       # initial condition: True = randomize the position and orientation upon each reset
        
        # Debug (deprecated)
        debug_hist_steps = 14

        # [Pursuit-Evasion Game] for 'simple_weaving' agent policy only
        if agent_policy_type == 'simple_weaving':
            agent_turn_freq = [60, 60] # sample how long to turn (tsteps) from [min, max]
            agent_straight_freq = [150, 150] # sample how long to keep straight (tsteps) from [min, max]
            
        randomize_init_turn_dir = False  # if True, then initial turn going left or right is randomized
        randomize_ctrl_bounds = False # if True, then randomize the maximum control authority based on the ranges below
        randomize_turn_dir = False #True  # if True, then randomize the turn direction to be the opposite turn dir w.p. 70%, keep same turn dir w.p. 30%; Else, always switch from left to right
        max_vx_range = [2, 3]
        max_vang_range = [1, 2]

    class robot_sensing:
        filter_type = "kf" # type of filter, kf
        # fov = 6.28      # = 360, full FOV
        fov = 1.54    # = 88 degrees, ZED 2 HD1080
    class terrain:
        mesh_type = 'plane'
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        # obstacle terrain only:
        fov_measure_heights = False

        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1m x 1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        terrain_length = 10
        terrain_width = 10
        num_rows = 3  # number of terrain rows (levels)
        num_cols = 3  # number of terrain cols (types)

        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete obstacles, stepping stones, forest]
        terrain_proportions = [0., 0., 0., 0., 0., 0., 1.]

        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

        # forest terrain type only:
        num_obstacles = 0  # number of "trees" in the environment
        obstacle_height = 0  # in [units]; e.g. 500 is very tall, 20 is managable by robot

    class commands: # note: commands and actions are the same for the high-level policy
        # num_robot_commands = 4        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        heading_command = False         # if true: compute ang vel command from heading error
        command_clipping = True        # if true: clip robot + agent commands to the ranges below
        use_joypad = False
        cmds_curriculum = False
        # At any of those, it will increase the predator speed by 20%
        curriculum_iters = [1800, 2000, 2400, 2800, 3100, 3800]
        class ranges: # clipping for agent (evader) and robot (pursuer)
            lin_vel_x = [-0, 3] # min max [m/s]
            lin_vel_y = [-1, 1] # min max [m/s]
            ang_vel_yaw = [-2, 2] # min max [rad/s]
            heading = [-3.14, 3.14]
            agent_lin_vel_x = [0, 2.5] # min max [m/s]
            agent_lin_vel_y = [-1, 1] # min max [m/s]
            agent_ang_vel_yaw = [-2, 2] # min max [rad/s]
            max_lin_speed = 3
            max_ang_vel = 3.14
            
    class init_state:
        # do not touch any of these configs
        agent_pos = [0.0, 0.0, 0.3] # x, y, z (agent pos)
        agent_rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        agent_lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        agent_ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        robot_pos = [0.0, 0.0, 0.42]  # x,y,z [m] (robot pos)
        robot_rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        robot_lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        robot_ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        curriculum = False
        max_init_dists = [2.0, 4.0, 6.0, 8.0, 10.0]

    class domain_rand:
        # do not touch any of these configs
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards_robot: # Robot (pursuer)!
        only_positive_rewards = False
        class scales:
            pursuit = -1.0
            exp_pursuit = 0. 
            command_norm = -0.0
            robot_foveation = 0.0
            robot_ang_vel = -0.0
            path_progress = 0.0
            time_elapsed = -0.0
            termination = 100.0

    class rewards_agent: # Evader! (Used for MARL)
        only_positive_rewards = False
        class scales:
            evasion = 2
            termination = -80.0

    class normalization:
        class obs_scales:
            height_measurements = 5.0

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [6, 0, 6]  # [m]
        lookat = [0, 0, 0]

    class sim:
        # Don't change this configs
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**15 #2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class DecHighLevelGameCfgPPO( BaseConfig ):
    seed = 1
    runner_class_name = 'DecGamePolicyRunner' # 'What is this?

    class policy:
        robot_policy_type = DecHighLevelGameCfg.env.robot_policy_type
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        privilege_enc_hidden_dims = [512, 256, 128]  # i.e. encoder_hidden_dims
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        eval_time = False # Used for some internal logic for test-time training. Do not change here.


        # Due to the complexity of Isaac Gym, some flags need to be repeated.
        future_len = DecHighLevelGameCfg.env.num_pred_steps
        history_len = DecHighLevelGameCfg.env.num_hist_steps
        num_robot_states = DecHighLevelGameCfg.env.num_robot_states
        num_priv_robot_states = DecHighLevelGameCfg.env.num_priv_robot_states
        num_robot_actions = DecHighLevelGameCfg.env.num_actions_robot
        num_agent_states = DecHighLevelGameCfg.env.num_agent_states
        
        num_latent = 8          # i.e., embedding size!
        
        if robot_policy_type == 'prediction_phase1':
            use_estimator = False   # True uses the learned estimator: zhat = E(x^history, uR^history)
            use_privilege_enc = True         # True uses the teacher estimator: z* = T(x^future)
            init_noise_std = 0.5
            num_privilege_enc_obs = num_robot_states * future_len  # i.e., 8-step future relative state
            num_estimator_obs = None
        elif robot_policy_type == 'prediction_phase2' or robot_policy_type == 'po_prediction_phase2':
            use_estimator = True   # True uses the learned estimator: zhat = E(x^history, uR^history)
            use_privilege_enc = True         # True uses the teacher estimator: z* = T(x^future)
            init_noise_std = 0.01
            num_privilege_enc_obs = num_priv_robot_states * future_len  # i.e., 8-step future relative state
            num_estimator_obs = num_robot_states * (history_len + 1) + num_robot_actions * history_len    # i.e., 8-step past rel-state and robot controls + present state

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0. #0.01, quite uncommon in RL, but seems to do well
        bc_coef = 0.
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive (uses KL divergence to compute learning rate), fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.


    class runner:
        # Set which agent to start training
        agent_id_to_train = 1 # 0 == evader, 1 == pursuer

        policy_class_name = 'ActorCriticGamesRMA'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 10         # per iteration. This implies 10*0.2 = 2 seconds per env
        max_iterations = 4001           # number of policy updates
        max_evolutions = 1            # (unused at the moment) number of times the two agents alternate policy updates (e.g., if 100, then each agent gets to be updated 50 times)

        # logging
        save_learn_interval = 200  # check for potential saves every this many iterations
        save_evol_interval = 1
        
        # load and resume
        resume_robot = False # Load a previous pursuer policy
        resume_agent = False # Load a previous evader policy
        load_run_agent = ''
        load_run_robot = ''
        evol_checkpoint_robot = 0  # Do not change this!
        learn_checkpoint_robot = 4000   # checkpoint number for pursuer. -1 = last saved model
        evol_checkpoint_agent = 0
        learn_checkpoint_agent = 2000 # checkpoint number for evader.
        resume_path = None  # updated from load_run and chkpt
        run_name = Run_Name
        experiment_name = Save_Folder_Name
