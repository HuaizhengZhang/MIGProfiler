from torch import nn
import torchvision.models as models
from transformers import AutoModelForSequenceClassification

from mig_perf.utils.common import model_names


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_cv_model(model_name) -> (nn.Module, int):
    """get models from https://pytorch.org/hub/.

    Args:
        model_name: model architecture.
    Return:
        model, input size.
    """
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    """ Vision Transformer base 16
    """
    if model_name == "vision_transformer":
        model_ft = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
        input_size = 224
    elif model_name == "resnet50":
        model_ft = models.resnet50(weights="IMAGENET1K_V1")
        input_size = 224
    elif model_name == "swin_transformer":
        model_ft = models.swin_b(weights='Swin_B_Weights.IMAGENET1K_V1')
        input_size = 224
    else:
        raise NotImplementedError(
            f"{model_name} is not supported, try 'vision_transformer' or 'resnet50', or 'swin_transformer'.")
    return model_ft, input_size


def load_nlp_model(model_name) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(model_names[model_name])

    return model
