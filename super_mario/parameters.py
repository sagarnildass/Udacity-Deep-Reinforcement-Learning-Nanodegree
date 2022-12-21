class opt:
    def __init__(self):
        self.world = 1
        self.stage = 1
        self.action_type = 'complex'
        self.lr = 1e-4
        self.gamma = 0.9
        self.tau = 1.0
        self.beta = 0.01
        self.num_local_steps = 50
        self.num_global_steps = 5e6
        self.num_processes = 12
        self.save_interval = 500
        self.max_actions = 200
        self.log_path = "tensorboard/a3c_super_mario_bros"
        self.saved_path = "trained_models"
        self.load_from_previous_stage = False
        self.use_gpu = True
