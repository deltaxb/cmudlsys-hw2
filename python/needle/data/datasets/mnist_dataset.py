from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        import gzip, struct
        # super().__init__(transforms)
        self.transforms = transforms
        with gzip.open(image_filename) as imf:
            magic = struct.unpack(">i", imf.read(4))[0]
            num_image = struct.unpack(">i", imf.read(4))[0]
            height = struct.unpack(">i", imf.read(4))[0]
            weight = struct.unpack(">i", imf.read(4))[0]
            X = np.frombuffer(imf.read(), np.uint8)
            assert X.size == num_image * height * weight
            X = X.reshape(num_image, height, weight, 1).astype(np.float32) / 255.0
        with gzip.open(label_filename) as laf:
            magic = struct.unpack(">i", laf.read(4))[0]
            num_image = struct.unpack(">i", laf.read(4))[0]
            y = np.frombuffer(laf.read(), np.uint8)
            assert y.size == num_image
        self.image = X
        self.label = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.image[index]
        # print(img.shape)
        if (self.transforms is not None):
            for transfrom in self.transforms:
                img = transfrom(img)
        return (img.reshape(img.shape[0], -1), self.label[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.label.shape[0]
        ### END YOUR SOLUTION