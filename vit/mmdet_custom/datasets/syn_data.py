import torch
import numpy as np

from pathlib import Path
from typing import List
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset
from mmcv.parallel.data_container import DataContainer
from mmdet.core.mask import BitmapMasks


@DATASETS.register_module()
class SynDataset(CustomDataset):
    CLASSES = CocoDataset.CLASSES

    PALETTE = CocoDataset.PALETTE

    def __init__(self, syn_dir):
        assert syn_dir != "", (
            "you should set the syn directory in the commandline interface."
        )

        pth_files = [
            x for x in Path(syn_dir).iterdir() if x.name.endswith(".pt")
        ]

        def psaq_collect(ckpt) -> List[dict]:
            total_samples = len(ckpt)
            import numpy as np

            rng = np.random.RandomState(0)
            masks = (rng.rand(1, 640, 640) > 0.1).astype(np.uint8)
            return [
                {
                    "img_metas": DataContainer(
                        {
                            "img_shape": (640, 640, 3),
                            "pad_shape": (640, 640, 3),
                        },
                        cpu_only=True,
                        padding_value=0,
                        stack=False,
                        pad_dims=2,
                    ),
                    "img": DataContainer(ckpt[i], stack=True),
                    "gt_bboxes": DataContainer(
                        torch.Tensor([[0.0000, 25.8699, 189.7085, 158.4098]])
                    ),
                    "gt_labels": DataContainer(
                        torch.zeros(1).type(torch.int64)
                    ),
                    "gt_masks": DataContainer(
                        BitmapMasks(masks, 640, 640), cpu_only=True
                    ),
                }
                for i in range(total_samples)
            ]

        def split_ckpt(ckpt) -> List[dict]:
            ckpt = torch.load(ckpt, map_location="cpu")
            # for PSAQ, we randomly generate the label
            if isinstance(ckpt, torch.Tensor):
                return psaq_collect(ckpt)
            total_samples = len(ckpt["img_metas"].data[0])
            return [
                {
                    "img_metas": DataContainer(
                        ckpt["img_metas"].data[0][i],
                        cpu_only=True,
                        padding_value=0,
                        stack=False,
                        pad_dims=2,
                    ),
                    "img": DataContainer(ckpt["img"][i], stack=True),
                    "gt_bboxes": DataContainer(ckpt["gt_bboxes"].data[0][i]),
                    "gt_labels": DataContainer(ckpt["gt_labels"].data[0][i]),
                    "gt_masks": DataContainer(
                        ckpt["gt_masks"].data[0][i], cpu_only=True
                    ),
                }
                for i in range(total_samples)
            ]

        self.samples = []
        for pth in pth_files:
            self.samples.extend(split_ckpt(pth))
        self.dataset_size = len(self.samples)
        print(f"loaded synthesized data {self.dataset_size}")

        self.flag = self.assign_flags()

    def assign_flags(self):
        flags = np.zeros(self.dataset_size, dtype=np.uint8)
        for i, sample in enumerate(self.samples):
            (h, w, _) = sample["img_metas"].data["img_shape"]
            if w / h > 1:
                flags[i] = 1
        return flags

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.samples[idx]
