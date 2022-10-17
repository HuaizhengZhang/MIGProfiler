from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
from torchvision import transforms, datasets
from transformers import AutoTokenizer

from mig_perf.utils.common import model_names


def load_places365_data(input_size, data_path, batch_size, num_workers=4) -> DataLoader:
    """transform data and load data into dataloader. Images should be arranged in this way by default: ::
        root/my_dataset/dog/xxx.png
        root/my_dataset/dog/xxy.png
        root/my_dataset/dog/[...]/xxz.png
        root/my_dataset/cat/123.png
        root/my_dataset/cat/nsdf3.png
        root/my_dataset/cat/[...]/asd932_.png
    Args:
        input_size (int): transformed image resolution, such as 224.
        data_path (string): eg. root/my_dataset/
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

    # Create traininG dataset
    image_dataset = datasets.ImageFolder(data_path, data_transform)
    # Create training and validation dataloaders
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    return dataloader


def _collate_fn(x, tokenizer, seq_length):
    x = default_collate(x)
    ret = tokenizer(x['review_body'], return_tensors='pt', padding=True, truncation=True, max_length=seq_length)
    return ret


def load_amazaon_review_data(model_name, data_path, seq_length, batch_size, num_workers=4):
    model_name = model_names[model_name]
    # prepare test data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_data, _ = load_dataset("amazon_reviews_multi", "all_languages", split=['train', 'test'], cache_dir=data_path)
    dataloader = DataLoader(test_data, batch_size=batch_size,
                            collate_fn=lambda x: _collate_fn(x, tokenizer, seq_length),
                            num_workers=num_workers)
    return dataloader
