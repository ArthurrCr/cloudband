import numpy as np
import torch

from cloudband.train.predictor import build_predictor

STACK_CHANNELS = 13
PATCH_SIZE = 32
OUTPUT_CLASSES = 4


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, OUTPUT_CLASSES, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_predictor_output_shape_matches_annotation_shape():
    predictor = build_predictor(TinySegmenter())
    stack = np.random.randint(0, 10000, size=(STACK_CHANNELS, PATCH_SIZE, PATCH_SIZE))

    prediction = predictor(stack)

    assert prediction.shape == (PATCH_SIZE, PATCH_SIZE)


def test_predictor_output_values_are_valid_class_indices():
    predictor = build_predictor(TinySegmenter())
    stack = np.random.randint(0, 10000, size=(STACK_CHANNELS, PATCH_SIZE, PATCH_SIZE))

    prediction = predictor(stack)

    assert prediction.min() >= 0
    assert prediction.max() < OUTPUT_CLASSES


def test_predictor_only_feeds_three_channels_to_the_model():
    seen_channels = []

    class ChannelRecordingModel(torch.nn.Module):
        def forward(self, x):
            seen_channels.append(x.shape[1])
            return torch.zeros(x.shape[0], OUTPUT_CLASSES, x.shape[2], x.shape[3])

    predictor = build_predictor(ChannelRecordingModel())
    stack = np.random.randint(0, 10000, size=(STACK_CHANNELS, PATCH_SIZE, PATCH_SIZE))
    predictor(stack)

    assert seen_channels == [3]


def test_predictor_normalizes_before_predicting():
    seen_ranges = []

    class RangeRecordingModel(torch.nn.Module):
        def forward(self, x):
            seen_ranges.append((x.min().item(), x.max().item()))
            return torch.zeros(x.shape[0], OUTPUT_CLASSES, x.shape[2], x.shape[3])

    predictor = build_predictor(RangeRecordingModel())
    stack = np.full((STACK_CHANNELS, PATCH_SIZE, PATCH_SIZE), 5000, dtype=np.int32)
    stack[3] = np.random.randint(0, 10000, size=(PATCH_SIZE, PATCH_SIZE))
    predictor(stack)

    low, high = seen_ranges[0]
    assert not (low >= 4999 and high <= 5001)