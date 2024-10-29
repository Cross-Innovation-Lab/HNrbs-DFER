
import os
from numpy.random import randint
from torch.utils import data
import torch
import glob
import os
import numpy as np
import csv
import PIL.Image as Image
import torchvision

from .video_transform import *

class DFEWDataset(data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.path = self.args.dataset.DFEW.train_dataset if mode == "train" else self.args.dataset.DFEW.test_dataset
        self.num_frames = self.args.dataset.num_frames
        self.image_size = self.args.transform.image_size
        self.mode = mode
        self.transform = self.get_transform()
        self.data = self.get_data()
            
        pass

    def get_data(self):
        full_data = []

        npy_path = self.path.replace('csv', 'npy')
        if os.path.exists(npy_path):
            full_data = np.load(npy_path, allow_pickle=True)
        else:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    path = row[0]
                    emotion = int(row[1]) - 1
                    while len(path) < 5:
                        path = "0" + path
                    path = os.path.join(self.args.dataset.root, "Clip/clip_224x224/clip_224x224/", path)
                    full_num_frames = len(os.listdir(path))

                    full_video_frames_paths = glob.glob(os.path.join(path, '*.jpg'))
                    full_video_frames_paths.sort()

                    full_data.append({"path": full_video_frames_paths, "emotion": emotion, "num_frames": full_num_frames})
        
                np.save(npy_path, full_data)
        
        return full_data

    def get_transform(self):

        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([GroupRandomSizedCrop(self.image_size),
                                                        GroupRandomHorizontalFlip(),
                                                        GroupColorJitter(self.args.augment.color_jitter),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        elif self.mode == "test":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                            Stack(),
                                                            ToTorchFormatTensor()])
        
        return transform

    def __getitem__(self, index):
        data = self.data[index]

        full_video_frames_paths = data['path']

        video_frames_paths = []
        full_num_frames = len(full_video_frames_paths)
        if self.num_frames == 0:
            self.num_frames = full_num_frames
        for i in range(self.num_frames):

            frame = int(full_num_frames * i / self.num_frames)
            if self.args.augment.random_sample:
                frame += int(random.random() * full_num_frames / self.num_frames)
                frame = min(full_num_frames - 1, frame)
            video_frames_paths.append(full_video_frames_paths[frame])

        images = []
        if self.args.dataset.merge == 1:
            for video_frames_path in video_frames_paths:
                images.append(Image.open(video_frames_path).convert('RGB'))
        else:
            _image = []
            for video_frames_path in video_frames_paths:
                if len(_image) == self.args.dataset.merge:
                    _image = np.asarray(_image)
                    _image = np.mean(_image, axis=0)
                    images.append(Image.fromarray(_image.astype(np.uint8)))
                    _image = []
                _image.append(np.asarray(Image.open(video_frames_path).convert('RGB')))
            _image = np.asarray(_image)
            _image = np.mean(_image, axis=0)
            images.append(Image.fromarray(_image.astype(np.uint8)))

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        batch = {'images':images, 'targets':data['emotion'], 'idx': index}

        return batch

    def __len__(self):
        return len(self.data)



