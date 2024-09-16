from __future__ import annotations
from typing import Type, TypeVar
from dataclasses import dataclass, fields, MISSING, is_dataclass
import json
import pickle
from utils.path_utils import create_parent_directory

T = TypeVar("T", bound="ConfigBase")


@dataclass
class ConfigBase:
    @classmethod
    def load_json(cls: type[T], path: str) -> T:
        """
        Load the class from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            ConfigBase: The class loaded from the JSON file.

        """
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj

    def save_json(self, path: str):
        """
        Saves the class as JSON file.

        Args:
            path (str): The path to the output JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load_pickle(cls: type[T], path: str) -> T:
        """
        Load the class from a pickle file.

        Args:
            path (str): The path to the pickle file.

        Returns:
            The class loaded from the pickle file.
        """
        return pickle_load_object(path)

    def save_pickle(self, path: str) -> None:
        """
        Saves the class as a pickle file.

        Args:
            path (str): The path to the output pickle file.
        """
        pickle_save_object(self, path)


def print_initialization(cls, include_default: bool = True, init_fields_only: bool = True) -> str:
    """
    Print the initialization of a dataclass as a string
    """
    if not is_dataclass(cls):
        print(f"ERROR::{cls.__name__} is not a dataclass")
        return ""

    print(f"{cls.__name__}(")
    for field in fields(cls):
        if init_fields_only and field.init is False:
            continue

        is_default = not isinstance(field.default, type(MISSING))
        val = None
        if include_default and is_default:
            val = field.default

        if type(val) is str:
            val = f'f"{val}"'
        print(f"    {field.name} = {val}, # {field.type}")
    print(")")



def pickle_load_object(file_path: str):
    """
    Load an object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        The loaded object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If there is an error loading the object from the pickle file.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"file does not exist: {file_path}")
    except Exception as e:
        raise ValueError(f"error loading object from pickle file: {e}")


def pickle_save_object(obj, file_path: str):
    """
    Save an object to a pickle file.

    Args:
        obj: The object to be saved.
        file_path (str): The path to the pickle file.

    Raises:
        ValueError: If there is an error saving the object to the pickle file.
    """
    try:
        create_parent_directory(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise ValueError(f"error saving object to pickle file: {e}")
