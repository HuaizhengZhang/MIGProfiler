import os
from pathlib import Path

from torch.utils.data import DataLoader, default_collate

model_names = {
    'distil_v1': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
    'distil_v2': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'MiniLM': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'bert-base': 'bert-base-multilingual-cased',
}

DEFAULT_DATASET_ROOT = Path.home() / '.dataset'


def load_places365_data(
        batch_size, data_root=str(DEFAULT_DATASET_ROOT / 'places365_standard'),
        num_workers=os.cpu_count()
):
    """transform data and load data into dataloader. Images should be arranged in this way by default: ::
        root/my_dataset/dog/xxx.png
        root/my_dataset/dog/xxy.png
        root/my_dataset/dog/[...]/xxz.png
        root/my_dataset/cat/123.png
        root/my_dataset/cat/nsdf3.png
        root/my_dataset/cat/[...]/asd932_.png
    Args:
        batch_size (int): batch size
        data_root (str): eg. root/my_dataset/
        num_workers (int): number of pytorch DataLoader worker subprocess
    """
    from torchvision import transforms, datasets

    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def load_amazon_review_data(tokenizer, batch_size, max_seq_len, num_workers=os.cpu_count()):
    from datasets import load_dataset

    # prepare test data
    train_dataset, val_dataset = load_dataset("amazon_reviews_multi", "all_languages", split=['train', 'test'])

    def collate_fn(x):
        x = default_collate(x)
        inputs = tokenizer(
            x['review_body'], padding=True, truncation=True,
            max_length=max_seq_len, return_tensors='pt',
        )
        inputs['labels'] = (x['stars'] - 1).long()
        return inputs

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_dataloader, val_dataloader

