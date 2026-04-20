from dotenv import load_dotenv
import os
import torch

# Function to get a unique directory name.
def get_unique_dir(base_path):
    counter = 1
    unique_path = base_path
    while os.path.exists(unique_path):
        unique_path = f"{base_path.rstrip('/')}_{counter}/"
        counter += 1
    return unique_path

class Config:
    def __init__(self, env_path: str, out_subpath: str):
        load_dotenv(env_path, override=True)
        
        # Set up output directory, if it exists, create a new one with a different name.
        if out_subpath is not None:
            self.out_subpath = get_unique_dir(out_subpath)
            os.makedirs(self.out_subpath)

            self.checkpoint_path = get_unique_dir(os.path.join(self.out_subpath, "checkpoints/"))
            self.checkpoint_path_sa = os.path.join(self.checkpoint_path, "sa/")
            self.checkpoint_path_cg = os.path.join(self.checkpoint_path, "cg/")
            os.makedirs(self.checkpoint_path)
            os.makedirs(self.checkpoint_path_sa)
            os.makedirs(self.checkpoint_path_cg)
        else:
            self.out_subpath = None
            self.checkpoint_path = None
            
        # Copy .env file to out_subpath
        if out_subpath is not None:
            with open(env_path, 'r') as src, open(os.path.join(self.out_subpath, '.env'), 'w') as dst:
                dst.write(src.read())
                
        self._load_base_config()
        self._load_SA_config()
        self._load_dataset_config()
        self._load_consensus_game_config()
        self._load_rnn_config()
        self._load_classifier_config()
        self._load_trainer_config()
        self._load_player_config()
        self._load_evaluator_config()
    
    def _load_base_config(self):
        self.device = os.getenv("DEVICE", "cuda")
        self.use_gpu = self.device == "cuda" and torch.cuda.is_available()
        self.tqdm_interval = 1 if os.environ.get("IS_NOHUP") is None else 60
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
    
    def _load_SA_config(self):
        # Slot Attention Model
        self.width = int(os.getenv("SA_WIDTH", 64))
        self.num_slots = int(os.getenv("SA_NUM_SLOTS", 11))
        self.slot_dim = int(os.getenv("SA_SLOT_DIM", 64))
        self.routing_iters = int(os.getenv("SA_ROUTING_ITERS", 3))
        # Slot Attention Training
        self.exp_name = os.getenv("SA_EXP_NAME", "default")
        self.seed = int(os.getenv("SA_SEED", 42))
        self.learning_rate = float(os.getenv("SA_LEARNING_RATE", 2e-4))
        self.weight_decay = float(os.getenv("SA_WEIGHT_DECAY", 1e-6))
        self.deterministic = os.getenv("SA_DETERMINISTIC", "True").lower() == "true"

    def _load_dataset_config(self):
        self.dataset = os.getenv("DATASET", "ch")
        self.dataset_path = os.getenv("DATASET_PATH", "./data")
        self.dataset_max_num_obj = int(os.getenv("DATASET_MAX_NUM_OBJ", 10))
        self.dataset_image_resolution = int(os.getenv("DATASET_IMAGE_RESOLUTION", 64))
        self.dataset_cache =  os.getenv("DATASET_CACHE", "False").lower() == "true"
        self.dataset_batch_size = int(os.getenv("DATASET_BATCH_SIZE", 64))
        self.dataset_eval_batch_size = int(os.getenv("DATASET_EVAL_BATCH_SIZE", 1))
        self.dataset_num_workers = int(os.getenv("DATASET_NUM_WORKERS", 16))
        self.labels = [int(label) for label in os.getenv("LABELS", "[0,1,2,3,4,5,6]").strip("[]").split(",")]

    def _load_consensus_game_config(self):
        self.n_players = int(os.getenv("N_PLAYERS", 2))
        self.consensus_game_length = int(os.getenv("CONSENSUS_GAME_LENGTH", 5))
        self.penalise_selections = os.getenv("PENALISE_SELECTIONS", "False").lower() == "true"
        self.penalise_selections_weight = float(os.getenv("PENALISE_SELECTIONS_WEIGHT", 0.1))
        self.relative_confidence_change = os.getenv("RELATIVE_CONFIDENCE_CHANGE", "False").lower() == "true"
        self.end_condition = os.getenv("END_CONDITION", "consensus")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
        self.always_classify = os.getenv("ALWAYS_CLASSIFY", "False").lower() == "true"
        self.second_selection_uniqueness_factor = float(os.getenv("SECOND_SELECTION_UNIQUENESS_FACTOR", 0.05))
        self.repeat_selection_reward_factor = float(os.getenv("REPEAT_SELECTION_REWARD_FACTOR", 0.05))
        self.sparse_reward_factor = float(os.getenv("SPARSE_REWARD_FACTOR", 0))
        self.informative_repeat_scaling = float(os.getenv("INFORMATIVE_REPEAT_SCALING", 0.05))
        self.per_agent_uniqueness = os.getenv("PER_AGENT_UNIQUENESS", "False").lower() == "true"

    def _load_rnn_config(self):
        self.rnn_input_size = int(os.getenv("RNN_INPUT_SIZE", 128))
        self.rnn_output_size = int(os.getenv("RNN_OUTPUT_SIZE", 128))

    def _load_classifier_config(self):
        self.classifier_hidden = int(os.getenv("CLASSIFIER_HIDDEN", 256))

    def _load_trainer_config(self):
        self.epochs = int(os.getenv("EPOCHS", 500))
        self.checkpoint = int(os.getenv("CHECKPOINT", 4))
        self.max_norm = float(os.getenv("MAX_NORM", -1.0))
        self.loss_threshold = float(os.getenv("LOSS_THRESHOLD", 1.0))
        self.reward_threshold = float(os.getenv("REWARD_THRESHOLD", 5.0))
        self.checkpoint_name = os.getenv("CHECKPOINT_NAME", "checkpoint")
        self.checkpoint_file_type = os.getenv("CHECKPOINT_FILE_TYPE", "pth")
        self.train_only_last_classification = os.getenv("TRAIN_ONLY_LAST_CLASSIFICATION", "False").lower() == "true"
        self.save_freq = int(os.getenv("SAVE_FREQ", 2))
        self.reinforce_loss_weight = float(os.getenv("REINFORCE_LOSS_WEIGHT", 1.0))
        self.ce_loss_weight = float(os.getenv("CE_LOSS_WEIGHT", 1.0))
        self.sa_loss_weight = float(os.getenv("SA_LOSS_WEIGHT", 1.0))
        self.em_reward_sa_loss_scaling = float(os.getenv("EM_REWARD_SA_LOSS_SCALING", 0.001))
        self.min_save_epochs = int(os.getenv("MIN_SAVE_EPOCHS", 100))

    def _load_player_config(self):
        self.player_gamma = float(os.getenv("PLAYER_GAMMA", 0.99))
        self.player_state_dim = self.rnn_output_size
        self.player_hidden_dim = int(os.getenv("PLAYER_HIDDEN_DIM", 128))
        self.player_action_dim = self.num_slots
        self.baseline_hidden_dim = int(os.getenv("BASELINE_HIDDEN_DIM", 128))
        self.player_lr = float(os.getenv("PLAYER_LR", 1e-4))
        self.entropy_beta = float(os.getenv("ENTROPY_BETA", 1e-3))
        self.baseline_beta = float(os.getenv("BASELINE_BETA", 0.5))
        self.use_baseline = self.baseline_beta > 0
        self.redundancy_lambda = float(os.getenv("REDUNDANCY_LAMBDA", 1e-1))
        self.consensus_lambda = float(os.getenv("CONSENSUS_LAMBDA", 1e-1))

    def _load_evaluator_config(self):
        self.eval_episodes = int(os.getenv("EVAL_EPISODES", 200))
        self.chart_frequency = int(os.getenv("CHART_FREQUENCY", 10))
        self.temp_tag = os.getenv("TEMP_TAG", "_temp")
        if self.out_subpath is not None:
            self.reward_temp_path = self.out_subpath + "rewards" + self.temp_tag + ".png"
            self.losses_temp_path = self.out_subpath + "losses" + self.temp_tag + ".png"
            self.results_temp_path = self.out_subpath + "results" + self.temp_tag + ".png"
            self.metrics_temp_path = self.out_subpath + "metrics" + self.temp_tag + ".png"
            self.visualisation_temp_path = self.out_subpath + "visualisation" + self.temp_tag + ".png"
            self.sa_visualisation_temp_path = self.out_subpath + "sa_visualisation" + self.temp_tag + ".png"
