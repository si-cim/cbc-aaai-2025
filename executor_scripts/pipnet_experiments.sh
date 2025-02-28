python deep_cbc/pipnet_executor.py --config-path=configs/cub_configs ++log_dir="'./experiment_logs/pipnet_cub_cnext26/run_seed_1'" ++seed=1 ++gpu_ids="'7'"
python deep_cbc/pipnet_executor.py --config-path=configs/cub_configs ++log_dir="'./experiment_logs/pipnet_cub_cnext26/run_seed_2'" ++seed=2 ++gpu_ids="'7'"
python deep_cbc/pipnet_executor.py --config-path=configs/cub_configs ++log_dir="'./experiment_logs/pipnet_cub_cnext26/run_seed_3'" ++seed=3 ++gpu_ids="'7'"
python deep_cbc/pipnet_executor.py --config-path=configs/cub_configs ++log_dir="'./experiment_logs/pipnet_cub_cnext26/run_seed_4'" ++seed=4 ++gpu_ids="'7'"
python deep_cbc/pipnet_executor.py --config-path=configs/cub_configs ++log_dir="'./experiment_logs/pipnet_cub_cnext26/run_seed_5'" ++seed=5 ++gpu_ids="'7'"

# Note 1: Override the "--config-path" argument to select the correct dataset and specify "++log_dir=" argument accordingly with correct dataset name for the given model.
# Note 2: Configuration files for PIPNet network corresponding to the ResNet50 network is not added with the hyperparameters.
# Note 3: If new configuration files are created for training different experiment configurations then ConfigStore file name also needs update in the pipnet_executor.py file.
