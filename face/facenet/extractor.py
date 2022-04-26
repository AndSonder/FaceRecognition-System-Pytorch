import torch
import torchvision.transforms as transforms
import cv2
from .model import Resnet34Triplet
import numpy as np
import os

class FaceExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        model_path, _ = os.path.split(os.path.realpath(__file__))
        model_path = os.path.join(model_path, "weights", "model_resnet34_triplet.pt")
        checkpoint = torch.load(model_path, map_location=device)
        self.model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=140),  # Pre-trained model uses 140x140 input images
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                # Normalization settings for the model, the calculated mean and std values
                std=[0.2457, 0.2175, 0.2129]  # for the RGB channels of the tightly-cropped glint360k face dataset
            )
        ])

    def extractor(self, img):
        img = np.asarray(img)
        img = self.preprocess(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        embedding = self.model(img)
        embedding = embedding.cpu().detach().numpy()
        return embedding
