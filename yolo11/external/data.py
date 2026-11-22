from copy import deepcopy
from pathlib import Path
from typing import Generator, List, Tuple
from pydantic import BaseModel
import torch

from ultralytics.utils import LOGGER  # type: ignore
from ultralytics.data.utils import check_det_dataset  # type: ignore
from ultralytics.data.dataset import YOLODataset  # type: ignore


class LabelBatch(BaseModel, arbitrary_types_allowed=True):
    """Data Transfer Object for converting raw detection labels to batch format.

    (arbitrary_types_allowed=True allows PyTorch tensor compatibility)

    Source Data Format:
    -------------------------------
    Each raw data sample contains:
        - im_file: str
            Path to original image file (used for naming generated pseudo-images)
        - ori_shape: Tuple[int, int]
            Original (height, width) of the image
        - resized_shape: Tuple[int, int]
            Image shape after model-input resizing
        - img: Any
            Original image data (not used in this pipeline)
        - cls: torch.Tensor
            Class labels for bounding boxes
        - bboxes: torch.Tensor
            Bounding box coordinates
        - batch_idx: torch.Tensor
            Index mapping between boxes and images:
                batch_idx[bbox_index] = image_index

            Note: len(bboxes) > len(im_file) because single image may contain multiple boxes.

    Transformed Structure:
    ----------------------------
        - sample_paths (aliased from im_file): List[str]
        - original_shapes (aliased from ori_shape): List[Tuple[int, int]]
        - resized_shapes: List[Tuple[int, int]]
        - classes: torch.Tensor (from cls)
        - bboxes: torch.Tensor
        - batch_indices: torch.Tensor (from batch_idx)

    Tensor Specifications:
        - classes: Shape [N], class IDs for N bounding boxes
        - bboxes: Shape [N, 4], format typically [x1, y1, x2, y2]
        - batch_indices: Shape [N], image index for each box
    """

    sample_paths: List[Path]
    original_shapes: List[List[int]]
    resized_shapes: List[List[int]]
    classes: torch.Tensor
    bboxes: torch.Tensor
    batch_indices: torch.Tensor

    def size(self) -> int:
        return len(self.sample_paths)


def to_yolo_format(label_batch: LabelBatch) -> dict:
    return {
        "cls": label_batch.classes,
        "bboxes": label_batch.bboxes,
        "batch_idx": label_batch.batch_indices,
    }


def merge_to_target_tensor(label_batch: LabelBatch) -> torch.Tensor:
    """
    To the format of [batch_idx, cls, centerx, centery, w, h]

    NOTICE: this function clones the current data.

    This is a hack to reuse current code.
    """
    return torch.cat(
        (
            label_batch.batch_indices.view(-1, 1),
            label_batch.classes.view(-1, 1),
            label_batch.bboxes,
        ),
        1,
    )


def from_target_tensor(label_batch: LabelBatch, target: torch.Tensor):
    """
    From [batch_idx, cls, centerx, centery, w, h] back.

    NOTICE: this function clones the target

    NOTICE: this function is in place
    """
    label_batch.batch_indices = target[:, 0].clone().detach()
    label_batch.classes = target[:, 1].clone().detach()
    label_batch.bboxes = target[:, 2:].clone().detach()


def _get_label_loader(
    calibration_size: int, yolo_dataloader: torch.utils.data.DataLoader
):
    for idx, data in enumerate(yolo_dataloader):
        if idx * (yolo_dataloader.batch_size or 1) >= calibration_size:
            return
        yield LabelBatch(
            sample_paths=data["im_file"],
            original_shapes=data["ori_shape"],
            resized_shapes=data["resized_shape"],
            classes=data["cls"],
            bboxes=data["bboxes"],
            batch_indices=data["batch_idx"],
        )


class LabelSet(BaseModel):
    nc: int
    names: dict[int, str]
    batch_size: int
    generator: Generator[LabelBatch, None, None]


def get_labelset_from_yolo(
    base_dataset_manifest: Path,
    batch_size: int,
    workers: int,
    calibration_size: int,
) -> LabelSet:
    dataset = check_det_dataset(base_dataset_manifest)
    yolo_dataset = YOLODataset(
        img_path=dataset["path"],
        data=dataset,
        task="detect",
        batch_size=batch_size,
        augment=False,
    )
    yolo_dataloader = torch.utils.data.DataLoader(
        dataset=yolo_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=getattr(yolo_dataset, "collate_fn", None),
    )
    generator = _get_label_loader(
        calibration_size=calibration_size, yolo_dataloader=yolo_dataloader
    )
    return LabelSet(
        nc=dataset["nc"],
        names=dataset["names"],
        batch_size=batch_size,
        generator=generator,
    )


def extract_label_from_batch(label: LabelBatch) -> List[LabelBatch]:
    samples = []
    for idx in range(len(label.sample_paths)):
        indices = label.batch_indices == idx
        samples.append(
            LabelBatch(
                sample_paths=[label.sample_paths[idx]],
                original_shapes=[label.original_shapes[idx]],
                resized_shapes=[label.resized_shapes[idx]],
                classes=label.classes[indices].unsqueeze(1),
                bboxes=label.bboxes[indices],
                batch_indices=torch.zeros(label.batch_indices[indices].shape),
            )
        )
    return samples


def collate_label_batches(ls: List[LabelBatch]) -> LabelBatch:
    sample_pathes = []
    original_shapes = []
    resized_shapes = []
    classes = []
    bboxes = []
    batch_indices = []

    for idx, l in enumerate(ls):
        sample_pathes.extend(l.sample_paths)
        original_shapes.extend(l.original_shapes)
        resized_shapes.extend(l.resized_shapes)
        classes.append(l.classes)
        bboxes.append(l.bboxes)
        batch_indices.append(l.batch_indices + idx)

    cls_tensor = torch.cat(classes, 0)
    bboxes_tensor = torch.cat(bboxes, 0)
    batch_indices_tensor = torch.cat(batch_indices, 0)

    return LabelBatch(
        sample_paths=sample_pathes,
        original_shapes=original_shapes,
        resized_shapes=resized_shapes,
        classes=cls_tensor,
        bboxes=bboxes_tensor,
        batch_indices=batch_indices_tensor,
    )


class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, label_set: List[LabelBatch]):
        super(LabelDataset, self).__init__()
        self.sample_list = []
        for label in label_set:
            self.sample_list.extend(extract_label_from_batch(label))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        return self.sample_list[idx]


# collect all fp-sampled labels in sampling_path, grouping them by [batch idx]
def _collect_sampled_labels(sampling_path: Path) -> List[Path]:
    return [file for file in sampling_path.iterdir()]


def get_sampled_label_loader(
    calibration_size: int,
    sampling_path: Path,
    batch_size: int,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    sampled_labels = _collect_sampled_labels(sampling_path)
    labels = []
    for idx, path in enumerate(sampled_labels):
        if idx * (batch_size or 1) > calibration_size:
            break
        sample_label: LabelBatch = torch.load(path, map_location=device)
        labels.append(sample_label)
    dataset = LabelDataset(labels)
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate_label_batches
    )


def get_sampled_labelset(
    sampling_label_path: Path,
    batch_size: int,
    calibration_size: int,
    nc: int,
    names: dict[int, str],
    device: torch.device,
) -> LabelSet:
    dataloader = get_sampled_label_loader(
        calibration_size=calibration_size,
        sampling_path=sampling_label_path,
        batch_size=batch_size,
        device=device,
    )
    return LabelSet(
        nc=nc, names=names, batch_size=batch_size, generator=dataloader
    )


# The generated pseudo data is arranged as follows:
#    weights /
#              images /
#              labels /
#


def extract_single_from_batch(
    img: torch.Tensor, label: LabelBatch
) -> List[dict]:
    samples = []
    for idx, img in enumerate(img):
        indices = label.batch_indices == idx
        samples.append(
            {
                "cls": label.classes[indices].unsqueeze(1),
                "bboxes": label.bboxes[indices],
                "batch_idx": torch.zeros(label.batch_indices[indices].shape),
                "img": img,
            }
        )
    return samples


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_list: List[Tuple[torch.Tensor, LabelBatch]],
        calibration_size: int,
    ):
        super(GeneratedDataset, self).__init__()
        self.sample_list = []
        for img, label in sample_list:
            self.sample_list.extend(extract_single_from_batch(img, label))
        LOGGER.info(f"selecting first {calibration_size} samples.")
        self.sample_list = self.sample_list[:calibration_size]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        return deepcopy(self.sample_list[idx])


def get_batch_limited_by_calibration(batches_path: Path) -> List[Path]:
    batch_pathes = [item for item in batches_path.iterdir()]
    return sorted(batch_pathes)


def build_pseudo_dataset(
    pseudo_image_list: List[Path],
    pseudo_label_list: List[Path],
    calibration_size: int,
) -> GeneratedDataset:
    sample_batches = [
        (
            torch.load(img_name, map_location="cpu"),
            torch.load(label_name, map_location="cpu"),
        )
        for img_name, label_name in zip(pseudo_image_list, pseudo_label_list)
    ]
    return GeneratedDataset(sample_batches, calibration_size=calibration_size)


def get_generated_dataset(
    pseudo_data_path: Path, calibration_size: int
) -> GeneratedDataset:
    pseudo_image_pathes = get_batch_limited_by_calibration(
        pseudo_data_path / "images"
    )
    pseudo_labels_pathes = get_batch_limited_by_calibration(
        pseudo_data_path / "labels"
    )
    return build_pseudo_dataset(
        pseudo_image_list=pseudo_image_pathes,
        pseudo_label_list=pseudo_labels_pathes,
        calibration_size=calibration_size,
    )


class GeneratedManifest(BaseModel):
    nc: int
    names: dict[int, str]
    batch_size: int


def save_gmanifest(dataset: LabelSet, save_dir: Path):
    manifest = GeneratedManifest(
        nc=dataset.nc,
        names=dataset.names,
        batch_size=dataset.batch_size,
    )
    json_path = save_dir / "manifest.json"
    with json_path.open("w") as f:
        f.write(manifest.model_dump_json(indent=2))


def load_gmanifest(sampling_weight_path: Path) -> GeneratedManifest:
    manifest_path = sampling_weight_path / "manifest.json"

    with manifest_path.open() as f:
        return GeneratedManifest.model_validate_json(f.read())
