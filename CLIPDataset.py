import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import typing as tp

class CLIPDataset(Dataset):
    def __init__(self, image_path, image_filenames, captions, tokenizer):
        """
        :image_path -- path to images
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        :tokenizer -- LM Tokenizer 
        """
        self.max_tokenizer_length = 200
        self.truncation = True
        self.padding = True
        self.image_path = image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.tokenizer = tokenizer
        self.encoded_captions = tokenizer(
            self.captions,
            padding=self.padding,
            max_length=self.max_tokenizer_length,
            truncation=self.truncation,
            return_tensors='pt'
        )
        self.transforms = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor()
        ])

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[torch.Tensor, str]]:

        """
        This one should return dict(keys=['image', 'caption'], value=[Image, Caption])
        """
        item = {
            key: values[idx] for key, values in self.encoded_captions.items()
        }
        full_image_path = Path(self.image_path) / Path(self.image_filenames[idx])
        image = Image.open(full_image_path).convert('RGB')
        image.load()
        item['image'] = self.transforms(image) 
        item['caption'] = self.captions[idx]
        return item


    def __len__(self):
        return len(self.captions)
