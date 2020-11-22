import os
import cv2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from utils import load_data, write_text
from flip import DatasetRetriever as FlipRetriever, Predictor as FlipPredictor, get_model as get_flip_model
from ocr import DatasetRetriever as OCRRetriever, Predictor as OCRPredictor, get_resnet34_model
from ctc_labeling import CTC_LABELING
from settings import *


def main():
    images, ids, rotates = load_data()
    ######
    # FLIP
    flip_images, flip_ids = [], []
    for image, _id, rotated in zip(images, ids, rotates):
        h, w, c = image.shape
        skip_flip = 0.8333 < (h / w) < 1.2
        if not skip_flip and rotated:
            flip_images.append(image)
            flip_ids.append(_id)

    flip_retriever = FlipRetriever(flip_ids, flip_images)
    flip_loader = DataLoader(
        flip_retriever,
        batch_size=BS,
        num_workers=NUM_WORKERS,
        sampler=SequentialSampler(flip_retriever),
        pin_memory=False,
    )
    flip_model = get_flip_model()
    flip_predictor = FlipPredictor(model=flip_model, device=DEVICE)
    flip_predictions = flip_predictor.run_inference(flip_loader)

    ocr_images, ocr_ids = [], []
    for image, _id in zip(images, ids):
        flip_prediction = flip_predictions.get(_id)
        if flip_prediction is not None:
            if flip_prediction['pred_vert']:
                image = cv2.flip(image, 0)
            if flip_prediction['pred_hori']:
                image = cv2.flip(image, 1)
        ocr_images.append(image)
        ocr_ids.append(_id)
    #####

    #####
    # OCR
    ocr_retriever = OCRRetriever(ocr_ids, ocr_images)
    ocr_loader = DataLoader(
        ocr_retriever,
        batch_size=BS,
        num_workers=NUM_WORKERS,
        sampler=SequentialSampler(ocr_retriever),
        pin_memory=False,
    )
    # ------------------------
    # RESNET34
    predictor = OCRPredictor(model=get_resnet34_model(), device=DEVICE)
    ocr_predictions = predictor.run_inference(ocr_loader)

    ######
    # WRITE RESULTS
    for ocr_prediction in ocr_predictions:
        file_path = os.path.join(OUT_FOLDER,  f'{ocr_prediction["id"]}.txt')
        write_text(file_path, CTC_LABELING.decode(ocr_prediction['raw_output']))
    ######


if __name__ == '__main__':
    main()
