import json

class BaseConfig(object):
    def __init__(self, config_file_path: str):
        with open(config_file_path) as json_file:
            self.json_dict = json.load(json_file)

class ModelConfig(BaseConfig):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)
        self.count_num_classes = self.json_dict['count_num_classes']
        self.num_classes_location = self.json_dict['num_classes_location']
        self.num_classes_class = self.json_dict['num_classes_class']
        self.num_filters = self.json_dict['num_filters']
        self.wd = self.json_dict['wd']
        

class TrainConfig(BaseConfig):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)
        self.batch_size = self.json_dict['batch_size']
        self.num_epochs = self.json_dict['num_epochs']
        self.max_lr = self.json_dict['max_lr']
        self.end_lr = self.json_dict['end_lr']
        self.decay_steps = self.json_dict['decay_steps']
        self.train_percentage = self.json_dict['train_percentage']
        self.test_percentage = self.json_dict['test_percentage']
        self.num_workers = self.json_dict['num_workers']
        self.train_on_gpu = self.json_dict['train_on_gpu']
        self.visible_gpus = self.json_dict['visible_gpus']
        self.checkpoint_folder = self.json_dict['checkpoint_folder']
        self.loss = self.json_dict['loss']
        self.metrics = self.json_dict['metrics']
        self.do_mixup = self.json_dict['do_mixup']
        self.epochs_restart = self.json_dict['epochs_restart']
        self.decay = self.json_dict['decay']
        self.momentum = self.json_dict['momentum']
        self.labeltype = self.json_dict['labeltype']
        self.ref_label_path_location = self.json_dict['ref_label_path_location']
        self.ref_label_path_class = self.json_dict['ref_label_path_class']
        self.resume_model = self.json_dict['resume_model']
        self.lr_scheduling_method = self.json_dict['lr_scheduling_method']
        self.do_data_preprocessing = self.json_dict['do_data_preprocessing']
        self.extract_ref_labels = self.json_dict['extract_ref_labels']
        
class DataConfig(BaseConfig):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)
        self.num_audio_channel = self.json_dict['num_audio_channel']
        self.duration = self.json_dict['duration']
        self.sampling_rate = self.json_dict['sampling_rate']
        self.num_freq_bin = self.json_dict['num_freq_bin']
        self.hop_length = self.json_dict['hop_length']
        self.num_fft = self.json_dict['num_fft']
        self.normalize = self.json_dict['normalize']
        self.mixup_alpha = self.json_dict['mixup_alpha']
        self.crop_percentage = self.json_dict['crop_percentage']
        self.source_dir_local = self.json_dict['source_dir_local']
        self.source_dir_training = self.json_dict['source_dir_training']
        self.valid_extension = self.json_dict['valid_extension']
        self.source_dir_augmented = self.json_dict['source_dir_augmented']
        self.source_dir_further_data = self.json_dict['source_dir_further_data']
        self.red_channels = self.json_dict['red_channels']
        self.augmented_data_folders = self.json_dict['augmented_data_folders']

class TestConfig(BaseConfig):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)
        self.path_test_data = self.json_dict['path_test_data']
        self.path_model = self.json_dict['path_model']
        self.path_model_class = self.json_dict['path_model_classtype']
        self.use_training_data = self.json_dict['use_training_data']
        self.savepath_dataframe = self.json_dict['savepath_dataframe']
        self.confidence_threshold = self.json_dict['confidence_threshold']

