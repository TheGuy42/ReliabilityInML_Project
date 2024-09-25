from __future__ import annotations
import os
from dataclasses import dataclass, field, InitVar
from enum import Enum
import pandas as pd
import csv
from typing import List, Tuple
from PIL import Image

from utils.path_utils import join_paths, Files, create_directory
from utils.config_base import ConfigBase
from yolo.box_utils import BoxFormat, BoxConverter, BoxUtils


class BBOX_FORMAT(Enum):
    XYXY = 0  # x_min, y_min, x_max, y_max
    XYWH = 1  # x_min, y_min, width, height
    YOLO = 2  # x_center, y_center, width, height (normalized)


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_xyxy(cls, x_min: float, y_min: float, x_max: float, y_max: float):
        return cls(x_min, y_min, x_max, y_max)

    def to_yolo(
        self, image_width: int, image_height: int
    ) -> Tuple[float, float, float, float]:
        x_center = (self.x_min + self.x_max) / 2
        y_center = (self.y_min + self.y_max) / 2
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return (
            x_center / image_width,
            y_center / image_height,
            width / image_width,
            height / image_height,
        )


class YOLO_Annotation:
    annot_cols = ["class", "x_center", "y_center", "width", "height"]
    def __init__(self, path: str) -> None:
        self.path = path

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            delimiter=" ",
            names=YOLO_Annotation.annot_cols,
        )
        return df

    @staticmethod
    def absolute(df: pd.DataFrame, image_width: int, image_height: int) -> pd.DataFrame:
        x_c, y_c, w, h = BoxUtils.unpack(df["x_center", "y_center", "width", "height"].to_numpy())
        df["x_center"] = df["x_center"] * image_width
        df["y_center"] = df["y_center"] * image_height
        df["width"] = df["width"] * image_width
        df["height"] = df["height"] * image_height
        return df


class DatasetReader:
    def __init__(self, dataset_root: str, subset: str):
        self.dataset_root = dataset_root

        image_path = join_paths(self.dataset_root, "images", subset)
        label_path = join_paths(self.dataset_root, "labels", subset)

        self.image_path = image_path
        self.label_path = label_path

        self.images: Files = Files(image_path)
        self.labels: Files = Files(label_path)


class wheatReader:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data = self._read_csv()
        self.img_path = join_paths(os.path(csv_file).parent, "images")

    def _read_csv(self) -> List[Tuple[str, List[BoundingBox], int]]:
        data = []
        with open(self.csv_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_path = join_paths(self.img_path, row["image_name"])
                boxes_string = row["BoxesString"]
                domain = int(row["domain"])
                bounding_boxes = self._parse_boxes_string(boxes_string)
                data.append((image_path, bounding_boxes, domain))
        return data

    def _parse_boxes_string(self, boxes_string: str) -> List[BoundingBox]:
        boxes = []
        for box_str in boxes_string.split(";"):
            x_min, y_min, x_max, y_max = map(int, box_str.split())
            boxes.append(BoundingBox(x_min, y_min, x_max, y_max))
        return boxes

    def __iter__(self):
        for item in self.data:
            yield item

    def append(self, weat_reader: wheatReader):
        self.data.extend(weat_reader.data)


class YoloDataset:
    def __init__(self, root_path: str, classes: dict[int, str]):
        self.root_path = root_path
        if os.path.exists(self.root_path):
            raise ValueError(f"Directory {self.root_path} already exists.")

        self.classes = classes

        self.create_dataset_dir()
        self.yaml = Writer(self.root_path, "data.yaml")
        self.train_file = Writer(self.root_path, "train.txt")
        self.val_file = Writer(self.root_path, "val.txt")
        self.test_file = Writer(self.root_path, "test.txt")

    def create_dataset_dir(self):
        os.makedirs(self.root_path, exist_ok=False)
        os.makedirs(join_paths(self.root_path, "images"))
        os.makedirs(join_paths(self.root_path, "labels"))

    def build_train_set(self, csv_path: str):
        data = wheatReader(csv_path)
        for image_path, boxes, domain in data:
            os.symlink(
                image_path,
                join_paths(self.root_path, "images", os.path.basename(image_path)),
            )

        pass

    def add_sample(self, image_path: str, boxes: List[BoundingBox], domain: int):
        image_name = os.path.basename(image_path)

        label_name = os.path.basename(image_name).replace(".png", ".txt")
        label_path = join_paths(self.root_path, "labels", label_name)
        label = Writer(label_path)

        os.symlink(image_path, join_paths(self.root_path, "images", image_name))

        for box in boxes:
            x_center, y_center, width, height = box.to_yolo(
                *self.get_img_size(image_path)
            )
            label.write(f"{domain} {x_center} {y_center} {width} {height}\n")
        pass

    def get_img_size(self, image_path: str) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            return img.size


class Writer:
    def __init__(self, file_path: str, file_name: str, mode: str = "w"):
        self.file_path = file_path
        self.file_name = file_name
        self.full_path = join_paths(self.file_path, self.file_name)

        self.f = open(self.full_path, mode)

    def write(self, data: str):
        self.f.write(data)

    def writelines(self, data: list[str]):
        self.f.writelines(data)

    def read(self, line: int = -1):
        return self.f.read(line)

    def readlines(self, line: int = -1):
        return self.f.readlines(line)

    def flush(self):
        self.f.flush()

    def close(self):
        self.flush()
        self.f.close()

    def __del__(self):
        self.f.close()
