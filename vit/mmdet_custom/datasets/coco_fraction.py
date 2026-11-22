from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class CocoFraction(CocoDataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        dataset_size=None,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        file_client_args=dict(backend="disk"),
    ):
        super().__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt,
            file_client_args,
        )
        if dataset_size:
            self.dataset_size = dataset_size
            print(
                f"truncating dataset from {len(self.data_infos)} to {dataset_size}"
            )
            self.data_infos = self.data_infos[: self.dataset_size]
            self.flag = self.flag[: self.dataset_size]
