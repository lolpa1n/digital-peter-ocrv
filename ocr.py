import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models

import numpy as np
from settings import *


def make_img_padding_right(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1 = 0  # (max_w - img_w) // 2
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg


class DatasetRetriever(Dataset):

    def __init__(self, ids, images):
        self.ids = ids
        self.images = images

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        image = self.images[idx].copy()
        image = make_img_padding_right(image, IMAGE_H, IMAGE_W)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return {
            'id': _id,
            'image': image,
        }


class Predictor:

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def run_inference(self, data_loader):
        self.model.eval()
        inference_result = []
        for batch in data_loader:
            with torch.no_grad():
                inference_result += self.inference_one_batch(batch)
        return inference_result

    def inference_one_batch(self, batch):
        outputs = self.model(batch['image'].to(self.device, dtype=torch.float32))
        predictions = []
        for sample_id, output in zip(batch['id'], outputs):
            predictions.append({
                'id': sample_id,
                'raw_output': output.detach().argmax(1).cpu(),
            })
        return predictions


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class RecognitionModel(nn.Module):
    def __init__(self, feature_extractor, time_feature_count, lstm_hidden, lstm_len, n_class):
        super(RecognitionModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, n_class)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        return x


def get_resnet_backbone():
    m = models.resnet34()
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)

def get_resnet34_model():
    backbone = get_resnet_backbone()
    model = RecognitionModel(backbone, **OCR_MODEL_PARAMS)
    model = model.to(DEVICE)
    checkpoint = torch.load(OCR_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
