import os
import time
import json

import numpy as np
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_score_thresh=box_thresh,
        box_score_thresh=box_thresh,
    )

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 90  # 不包含背景
    box_thresh = 0.5
    weights_path = "./mask_rcnn_weights.pth"
    label_json_path = "./coco91_indices.json"
    ckpt_path = "<PLACEHOLDER>/mask_rcnn/Distill/Test/exp/weights/iter2500/"
    ckpts_list = [
        os.path.join(ckpt_path, item) for item in os.listdir(ckpt_path)
    ]
    kind = "generate"
    save_path = f"./predict/{kind}"

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(
        weights_path
    )
    weights_dict = torch.load(weights_path, map_location="cpu")
    weights_dict = (
        weights_dict["model"] if "model" in weights_dict else weights_dict
    )
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), (
        "json file {} dose not exist.".format(label_json_path)
    )
    with open(label_json_path, "r") as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(ckpt_path), f"{ckpt_path} does not exits."
    box_num = 0
    # original_img = Image.open(img_path).convert('RGB')
    for i, ckpt in enumerate(ckpts_list):
        ckpt = torch.load(ckpt, map_location=device)
        tensor = ckpt[kind].tensors
        bs = tensor.shape[0]
        # print(f'batch size is {bs}')
        for j in range(bs):
            img_num = bs * i + j
            img = ckpt[kind].tensors[j : j + 1, :, :, :]

            # # from pil image to tensor, do not normalize image
            # data_transform = transforms.Compose([transforms.ToTensor()])
            # img = data_transform(original_img)
            # # expand batch dimension
            # img = torch.unsqueeze(img, dim=0)

            model.eval()  # 进入验证模式
            with torch.no_grad():
                # init
                if i == 0 and j == 0:
                    img_height, img_width = img.shape[-2:]
                    init_img = torch.zeros(
                        (1, 3, img_height, img_width), device=device
                    )
                    model(init_img)

                # t_start = time_synchronized()
                predictions = model(img.to(device))[0]
                # t_end = time_synchronized()
                # print("inference+NMS time: {}".format(t_end - t_start))

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()
                predict_mask = np.squeeze(
                    predict_mask, axis=1
                )  # [batch, 1, h, w] -> [batch, h, w]

                if len(predict_boxes) == 0:
                    print("没有检测到任何目标!")
                    continue

                print(f"img shape is {img.shape}")
                print(f"boxes num is {len(predict_boxes)}")
                box_num += len(predict_boxes)
                to_pil_image = transforms.ToPILImage()
                image = to_pil_image(img.squeeze(0))

                plot_img = draw_objs(
                    image,
                    boxes=predict_boxes,
                    classes=predict_classes,
                    scores=predict_scores,
                    masks=predict_mask,
                    category_index=category_index,
                    line_thickness=3,
                    font="arial.ttf",
                    font_size=20,
                )
                # plt.imshow(plot_img)
                # plt.show()
                # 保存预测的图片结果
                plot_img.save(f"{save_path}/image{img_num}.jpg")
    print(f"total box num is {box_num}")


if __name__ == "__main__":
    main()
