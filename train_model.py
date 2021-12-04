import os
import torch
from torch.utils import data
from PIL import Image
# from torchvision import transforms
import utils
import transforms as T
import json
import numpy as np
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class COCODataset(data.Dataset):
    
    # initialise function of class
    def __init__(self, root, image_to_annotations, new_image_ids, transforms):
        # the data directory 
        self.root = root
        self.new_image_ids = new_image_ids
        self.image_to_annotations = image_to_annotations
        self.transforms = transforms

    # obtain the sample with the given index
    def __getitem__(self, index):
        print(index)
        img_filename = f'cam{int((self.new_image_ids[index] - (self.new_image_ids[index] % 10000)) / 10000)}_{self.new_image_ids[index] % 10000}.png'
        # load images and masks
        img_path = os.path.join(self.root, img_filename)
        annotations = self.image_to_annotations[self.new_image_ids[index]]
        img = Image.open(img_path).convert("RGB")

        # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # masks = mask == obj_ids[:, None, None]

        masks = []
        boxes = []
        labels = []
        areas = []
        for annotation in annotations:
            mask = np.zeros(img.size)
            contours = np.array(annotation['segmentation']).reshape(-1, 2)
            print(contours)
            if len(contours) == 0:
                print('bad!!')
            cv2.drawContours(mask, [contours], -1, color=1, thickness=cv2.FILLED)
            masks.append(mask)
            bbox = annotation['bounding_box']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(annotation['category_id'])
            areas.append(annotation['area'])


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        # TODO
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # the total number of samples (optional)
    def __len__(self):
        return len(self.image_to_annotations)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

if __name__ == '__main__':

    training_dir = os.path.join(os.path.dirname(__file__), 'training_data')
    filename = os.path.join(training_dir, "coco.json")

    num_iterations = 100
    with open(filename, 'r') as f:
        coco_json = json.load(f)

    image_to_annotations = {}
    for annotation in coco_json['annotations']:
        image_id = annotation['image_id']
        image_to_annotations.setdefault(image_id, []).append(annotation)

    new_image_ids = {}
    for i, image_id in enumerate(image_to_annotations):
        new_image_ids[i] = image_id

    assert len(image_to_annotations) == len(new_image_ids)
    print(image_to_annotations.keys())
    print(new_image_ids)

    model = get_model_instance_segmentation(3)

    dataset = COCODataset(training_dir, image_to_annotations, new_image_ids, get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    # For Training
    images,targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images,targets)   # Returns losses and detections
    print(output)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)           # Returns predictions

    print(predictions)


