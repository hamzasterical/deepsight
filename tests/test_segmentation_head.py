import pytest
import torch

from src.models.segmentation_head import DecoderBlock, SegmentationHead


# ==============================================================
# FIXTURES
# ==============================================================


@pytest.fixture
def fused_features():
    return torch.randn(2, 768, 7, 7)


@pytest.fixture
def skip_features():
    return [
        torch.randn(2, 112, 14, 14),
        torch.randn(2, 40, 28, 28),
        torch.randn(2, 24, 56, 56),
        torch.randn(2, 16, 112, 112),
    ]


# ==============================================================
# DECODER BLOCK
# ==============================================================


class TestDecoderBlockInit:
    def test_default_init(self):
        block = DecoderBlock(256, 128)
        assert block is not None
        assert isinstance(block.up, torch.nn.ConvTranspose2d)
        assert block.up.in_channels == 256
        assert block.up.out_channels == 128

    def test_with_skip_channels(self):
        block = DecoderBlock(256, 128, skip_channels=64)
        assert block.conv[0].in_channels == 128 + 64
        assert block.conv[0].out_channels == 128

    def test_without_batchnorm(self):
        block = DecoderBlock(256, 128, use_batchnorm=False)
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in block.conv)

    def test_with_batchnorm(self):
        block = DecoderBlock(256, 128, use_batchnorm=True)
        assert any(isinstance(m, torch.nn.BatchNorm2d) for m in block.conv)

    def test_upsample_doubles_spatial(self):
        block = DecoderBlock(256, 128)
        x = torch.randn(1, 256, 7, 7)
        out = block.up(x)
        assert out.shape[-2:] == (14, 14)


class TestDecoderBlockForward:
    def test_forward_without_skip(self):
        block = DecoderBlock(256, 128)
        x = torch.randn(2, 256, 7, 7)
        out = block(x)
        assert out.shape == (2, 128, 14, 14)

    def test_forward_with_skip(self):
        block = DecoderBlock(256, 128, skip_channels=32)
        x = torch.randn(2, 256, 7, 7)
        skip = torch.randn(2, 32, 14, 14)
        out = block(x, skip)
        assert out.shape == (2, 128, 14, 14)

    def test_forward_dtype(self):
        block = DecoderBlock(256, 128)
        x = torch.randn(2, 256, 7, 7)
        out = block(x)
        assert out.dtype == torch.float32

    def test_skip_spatial_mismatch_resized(self):
        block = DecoderBlock(256, 128, skip_channels=32)
        x = torch.randn(1, 256, 7, 7)
        skip = torch.randn(1, 32, 13, 13)
        out = block(x, skip)
        assert out.shape == (1, 128, 14, 14)


# ==============================================================
# SEGMENTATION HEAD
# ==============================================================


class TestSegmentationHeadInit:
    def test_default_init(self):
        head = SegmentationHead()
        assert head.fused_channels == 768
        assert head.decoder_channels == [256, 128, 64, 32, 16]
        assert head.skip_channels == [112, 40, 24, 16]
        assert len(head.blocks) == 5

    def test_custom_channels(self):
        head = SegmentationHead(
            fused_channels=256,
            decoder_channels=[128, 64, 32],
            skip_channels=[64, 32],
        )
        assert head.fused_channels == 256
        assert len(head.blocks) == 3

    def test_no_skip_connections(self):
        head = SegmentationHead(
            skip_channels=[],
            decoder_channels=[128, 64, 32, 16, 8],
        )
        assert len(head.blocks) == 5

    def test_invalid_lengths_raises(self):
        with pytest.raises(ValueError):
            SegmentationHead(decoder_channels=[128, 64], skip_channels=[32, 16])

    def test_final_conv_output_channels(self):
        head = SegmentationHead()
        assert head.final_conv.out_channels == 1
        assert head.final_conv.in_channels == 16

    def test_custom_output_channels(self):
        head = SegmentationHead(output_channels=3)
        assert head.final_conv.out_channels == 3

    def test_without_batchnorm(self):
        head = SegmentationHead(use_batchnorm=False)
        assert not any(
            isinstance(m, torch.nn.BatchNorm2d)
            for block in head.blocks
            for m in block.conv
        )


class TestSegmentationHeadForward:
    def test_forward_with_skips(self, fused_features, skip_features):
        head = SegmentationHead()
        head.eval()
        out = head(fused_features, skip_features)
        assert out.shape == (2, 1, 224, 224)

    def test_forward_without_skips(self, fused_features):
        head = SegmentationHead(skip_channels=[])
        head.eval()
        out = head(fused_features)
        assert out.shape == (2, 1, 224, 224)

    def test_forward_dtype(self, fused_features, skip_features):
        head = SegmentationHead()
        head.eval()
        out = head(fused_features, skip_features)
        assert out.dtype == torch.float32

    def test_output_is_sigmoid_probability(self, fused_features, skip_features):
        head = SegmentationHead()
        head.eval()
        out = head(fused_features, skip_features)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_sample(self):
        head = SegmentationHead()
        head.eval()
        fused = torch.randn(1, 768, 7, 7)
        skips = [torch.randn(1, ch, 14 // (2 ** i), 14 // (2 ** i)) for i, ch in enumerate([112, 40, 24, 16])]
        # Fix skip shapes: [14, 14], [28, 28], [56, 56], [112, 112]
        skips = [
            torch.randn(1, 112, 14, 14),
            torch.randn(1, 40, 28, 28),
            torch.randn(1, 24, 56, 56),
            torch.randn(1, 16, 112, 112),
        ]
        out = head(fused, skips)
        assert out.shape == (1, 1, 224, 224)

    def test_gradient_flow(self, fused_features, skip_features):
        head = SegmentationHead()
        fused_features.requires_grad_(True)
        for s in skip_features:
            s.requires_grad_(True)
        out = head(fused_features, skip_features)
        loss = out.sum()
        loss.backward()
        assert fused_features.grad is not None
        for s in skip_features:
            assert s.grad is not None

    def test_partial_skip_features(self, fused_features):
        head = SegmentationHead()
        head.eval()
        partial_skips = [torch.randn(2, 112, 14, 14)]
        out = head(fused_features, partial_skips)
        assert out.shape == (2, 1, 224, 224)

    def test_multiple_forward_passes(self, fused_features, skip_features):
        head = SegmentationHead()
        head.eval()
        out1 = head(fused_features, skip_features)
        out2 = head(fused_features, skip_features)
        assert torch.allclose(out1, out2)

    def test_large_batch(self):
        head = SegmentationHead()
        head.eval()
        fused = torch.randn(16, 768, 7, 7)
        skips = [
            torch.randn(16, 112, 14, 14),
            torch.randn(16, 40, 28, 28),
            torch.randn(16, 24, 56, 56),
            torch.randn(16, 16, 112, 112),
        ]
        out = head(fused, skips)
        assert out.shape == (16, 1, 224, 224)


class TestSegmentationHeadEdgeCases:
    def test_train_eval_mode(self):
        head = SegmentationHead()
        head.train()
        assert head.training is True
        head.eval()
        assert head.training is False

    def test_parameters_exist(self):
        head = SegmentationHead()
        params = list(head.parameters())
        assert len(params) > 0

    def test_wrong_fused_channels_raises(self):
        head = SegmentationHead()
        head.eval()
        fused = torch.randn(2, 256, 7, 7)
        with pytest.raises(Exception):
            head(fused)

    def test_fused_batch_mismatch_with_skip(self, fused_features):
        head = SegmentationHead()
        head.eval()
        skips = [torch.randn(4, 112, 14, 14)]
        with pytest.raises(RuntimeError):
            head(fused_features, skips)
