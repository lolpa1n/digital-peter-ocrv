import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models

import numpy as np

from utils import resize_if_need, make_img_padding_around
from settings import *


class DatasetRetriever(Dataset):
    def __init__(self, ids, images):
        self.ids = ids
        self.images = images

    def __getitem__(self, idx):
        _id = self.ids[idx]
        image = self.images[idx].copy()
        image = make_img_padding_around(image, IMAGE_H, IMAGE_W)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return {
            'id': _id,
            'image': image,
        }

    def __len__(self):
        return len(self.images)


class Predictor:

    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)

    def run_inference(self, data_loader):
        self.model.eval()
        inference_result = {}
        for batch in data_loader:
            with torch.no_grad():
                for prediction in self.inference_one_batch(batch):
                    inference_result[prediction['id']] = {
                        'pred_vert': prediction['pred_vert'],
                        'pred_hori': prediction['pred_hori'],
                    }
        return inference_result

    def inference_one_batch(self, batch):
        pred_vert, pred_hori = self.model(batch['image'].to(self.device, dtype=torch.float32))
        proba_vert = pred_vert[:, 1].cpu().numpy()
        proba_hori = pred_hori[:, 1].cpu().numpy()
        predictions = []
        for sample_id, vert, hori in zip(batch['id'], proba_vert, proba_hori):
            predictions.append({
                'id': sample_id,
                'pred_vert': vert > 0.7,
                'pred_hori': hori > 0.7,
            })
        return predictions


class FlipModel(nn.Module):
    def __init__(self):
        super(FlipModel, self).__init__()
        m = models.resnet34()
        self.backbone = nn.Sequential(*[m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool])
        self.vertical = nn.Linear(512, 2)
        self.horizontal = nn.Linear(512, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        vert = self.vertical(x)
        hori = self.horizontal(x)
        return vert, hori


def get_model():
    model = FlipModel()
    model = model.to(DEVICE)
    checkpoint = torch.load(FLIP_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
