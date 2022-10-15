from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
from torchvision import transforms, datasets
from transformers import AutoTokenizer

from utils.common import model_names


def load_imagenet_data(input_size, batch_size, num_workers) -> DataLoader:
    """download and wrap imagenet training set as Dataloader

    Args:
        input_size (int): transformed image resolution, such as 224.
        batch_size (int): batch size
        num_workers (int): number of pytorch DataLoader worker subprocess
    """
    data_transform = transforms.Compose([
        transforms.Resize(
            input_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = datasets.ImageNet(
        root="data",
        train=True,
        download=True,
        transform=data_transform
    )
    dataloader = DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return dataloader


def _collate_fn(x, tokenizer, seq_length):
    x = default_collate(x)
    ret = tokenizer(x['review_body'], return_tensors='pt', padding=True, truncation=True, max_length=seq_length)
    return ret


def load_amazaon_review_data(model_name, seq_length, batch_size, num_workers):
    model_name = model_names[model_name]
    # prepare test data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_data, _ = load_dataset("amazon_reviews_multi", "all_languages", split=['train', 'test'])
    dataloader = DataLoader(test_data, batch_size=batch_size,
                            collate_fn=lambda x: _collate_fn(x, tokenizer, seq_length),
                            num_workers=num_workers)
    return dataloader
