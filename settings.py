import os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

NUM_WORKERS = 4
BS = 32

IMG_FOLDER = '/data'
OUT_FOLDER = '/output'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####
# OCR
IMAGE_H = 128
IMAGE_W = 1024

OCR_CHARS = '* )+0123456789[]bdfghilmnrstабвгдежзийклмнопрстуфхцчшщъыьэюяѣ⊕⊗'
OCR_MODEL_PARAMS = {
    'time_feature_count': 256,
    'lstm_hidden': 256,
    'lstm_len': 3,
    'n_class': len(OCR_CHARS),
}
OCR_MODEL_PATH = os.path.join(ROOT, 'models', 'DIG211-best_cer.pt')
FLIP_MODEL_PATH = os.path.join(ROOT, 'models', 'flip_model.pt')
######
