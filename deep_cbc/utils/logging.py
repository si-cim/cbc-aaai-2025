# import os
import argparse
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

from deep_cbc.utils.args import save_args, save_config


class Log:
    """
    Object for managing the log directory.
    """

    def __init__(self, log_dir: Union[str, Path]):
        # Store log in log_dir path.
        self._log_dir = Path(log_dir)
        self._logs = dict()

        # Ensuring that below directories exist, otherwise create them.
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.metadata_dir.is_dir():
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if not self.checkpoint_dir.is_dir():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir / Path("checkpoints")

    @property
    def metadata_dir(self):
        return self._log_dir / Path("metadata")

    def log_message(self, msg: str):
        """
        Write a message to the log file.
        :param msg: the message string to be written to the log file.
        """
        file_path = self.log_dir / Path("log.txt")
        if not file_path.is_file():
            open(file_path, "w").close()  # Make log file empty if it already exists.
        with open(file_path, "a") as f:
            f.write(msg + "\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information.
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. epoch number).
        :param value_names: The names of the attributes that are logged.
        """
        if log_name in self._logs.keys():
            raise Exception("Log already exists!")
        # Add to existing logs.
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns.
        with open(Path(self.log_dir) / Path(f"{log_name}.csv"), "w") as f:
            f.write(",".join((key_name,) + value_names) + "\n")

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file.
        :param log_name: The name of the log file.
        :param key: The key attribute for logging these values.
        :param values: value attributes that will be stored in the log.
        """
        if log_name not in self._logs.keys():
            raise Exception("Log not existent!")
        if len(values) != len(self._logs[log_name][1]):
            raise Exception("Not all required values are logged!")
        # Write a new line with the given values.
        print(self.log_dir)
        log_csv_path = self.log_dir / Path(f"{log_name}.csv")
        print(log_csv_path)
        with open(log_csv_path, "a") as f:
            f.write(",".join(str(v) for v in (key,) + values) + "\n")

    def log_args(self, args: Union[argparse.Namespace, OmegaConf]):
        if isinstance(args, argparse.Namespace):
            save_args(args, self._log_dir)
        if isinstance(args, OmegaConf):
            save_config(args, self._log_dir)
