import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def get_emoji_loader(
    emoji_type: str,
    image_size: int,
    batch_size: int,
    num_workers: int
):

    """Returns a pair of DataLoaders for training and test sets"""

    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_path = os.path.join('./emojis', emoji_type, 'train')
    test_path  = os.path.join('./emojis', emoji_type, 'test')

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset  = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_dloader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return train_dloader, test_dloader
