"""Unit and integration tests for baseline audio preprocessing and model inference (Week 1, Task 6).

All tests use synthetic signals and synthetic audio fixtures strictly for
pipeline validation. These do NOT represent real dataset benchmark evaluations.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from prj55.audio import (
    TARGET_SAMPLES,
    AudioValidationError,
    load_audio,
    pad_or_truncate_waveform,
    preprocess_audio,
)
from prj55.models.aasist import AASIST, AASISTConfig, load_aasist_l


@pytest.fixture
def valid_flac_audio(tmp_path: Path) -> Path:
    """Creates a valid 16 kHz mono FLAC file."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    flac_path = tmp_path / "valid_sample.flac"
    sf.write(str(flac_path), data, sr, format="FLAC")
    return flac_path


@pytest.fixture
def stereo_flac_audio(tmp_path: Path) -> Path:
    """Creates a 2-channel (stereo) 16 kHz FLAC file."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    ch1 = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    ch2 = (0.3 * np.cos(2 * np.pi * 880.0 * t)).astype(np.float32)
    stereo_data = np.stack([ch1, ch2], axis=1)
    stereo_path = tmp_path / "stereo_sample.flac"
    sf.write(str(stereo_path), stereo_data, sr, format="FLAC")
    return stereo_path


@pytest.fixture
def wrong_sr_flac_audio(tmp_path: Path) -> Path:
    """Creates a FLAC file with incorrect sample rate (44.1 kHz)."""
    sr = 44100
    data = np.zeros(sr, dtype=np.float32)
    path = tmp_path / "wrong_sr.flac"
    sf.write(str(path), data, sr, format="FLAC")
    return path


class TestAudioInputPipeline:
    """Tests input audio loading, format verification, and edge-case handling."""

    def test_load_valid_flac(self, valid_flac_audio: Path):
        data = load_audio(valid_flac_audio)
        assert data.ndim == 1
        assert len(data) == 32000
        assert np.all(np.isfinite(data))

    def test_downmix_stereo_to_mono(self, stereo_flac_audio: Path):
        data = load_audio(stereo_flac_audio)
        assert data.ndim == 1
        assert len(data) == 16000

    def test_wrong_sample_rate_rejected(self, wrong_sr_flac_audio: Path):
        with pytest.raises(AudioValidationError, match="Sample rate mismatch"):
            load_audio(wrong_sr_flac_audio)

    def test_empty_audio_file_rejected(self, tmp_path: Path):
        empty_file = tmp_path / "empty.flac"
        empty_file.touch()
        with pytest.raises(AudioValidationError, match="empty"):
            load_audio(empty_file)

    def test_corrupt_audio_file_rejected(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupt.flac"
        corrupt_file.write_bytes(b"NOT_A_VALID_FLAC_HEADER_RANDOM_GARBAGE")
        with pytest.raises(AudioValidationError, match="Corrupt or unsupported"):
            load_audio(corrupt_file)

    def test_nonexistent_audio_file_raises(self, tmp_path: Path):
        absent = tmp_path / "non_existent.flac"
        with pytest.raises(FileNotFoundError):
            load_audio(absent)

    def test_pad_or_truncate_short_waveform(self):
        # 16000 samples (< 64600) should be repeated up to 64600
        short_wave = np.ones(16000, dtype=np.float32)
        tensor = pad_or_truncate_waveform(short_wave, target_length=TARGET_SAMPLES)
        assert tensor.shape == (1, TARGET_SAMPLES)
        assert torch.all(tensor == 1.0)

    def test_pad_or_truncate_long_waveform(self):
        # 80000 samples (> 64600) should be truncated to 64600
        long_wave = np.arange(80000, dtype=np.float32)
        tensor = pad_or_truncate_waveform(long_wave, target_length=TARGET_SAMPLES)
        assert tensor.shape == (1, TARGET_SAMPLES)
        assert tensor[0, 0] == 0.0
        assert tensor[0, -1] == 64599.0

    def test_pad_or_truncate_empty_waveform_raises(self):
        empty_wave = np.array([], dtype=np.float32)
        with pytest.raises(AudioValidationError, match="empty"):
            pad_or_truncate_waveform(empty_wave)

    def test_preprocess_audio_end_to_end(self, valid_flac_audio: Path):
        tensor = preprocess_audio(valid_flac_audio)
        assert tensor.shape == (1, TARGET_SAMPLES)
        assert tensor.dtype == torch.float32
        assert torch.all(torch.isfinite(tensor))


class TestModelInference:
    """Tests AASIST-L architecture execution, output shapes, and numerical stability."""

    def test_model_forward_pass_shapes(self):
        model = AASIST(AASISTConfig())
        model.eval()

        # Batch of 2 synthetic inputs
        synthetic_batch = torch.randn(2, TARGET_SAMPLES)
        with torch.no_grad():
            last_hidden, output = model(synthetic_batch)

        assert last_hidden.shape == (2, 160)
        assert output.shape == (2, 2)
        assert torch.all(torch.isfinite(last_hidden))
        assert torch.all(torch.isfinite(output))

    def test_load_aasist_l_with_verified_checkpoint(self):
        ckpt_path = Path("models/weights/AASIST-L.pth")
        if not ckpt_path.is_file():
            pytest.skip("AASIST-L checkpoint not downloaded locally")

        model = load_aasist_l(ckpt_path, device="cpu")
        model.eval()

        synthetic_input = torch.randn(1, TARGET_SAMPLES)
        with torch.no_grad():
            last_hidden, logits = model(synthetic_input)

        assert logits.shape == (1, 2)
        assert torch.all(torch.isfinite(logits))
