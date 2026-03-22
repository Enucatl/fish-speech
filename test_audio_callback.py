"""
Smoke test for AudioSampleCallback.

Builds a minimal mock Trainer/LightningModule from the real model,
then calls _generate_sample directly to verify the full pipeline:
  model load -> reference encode -> generate -> codec decode -> save wav
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
from loguru import logger

CHECKPOINT_PATH = "checkpoints/s2-pro"
CODEC_PATH = "checkpoints/s2-pro/codec.pth"
REFERENCE_AUDIO = "references/barbero/74_La_crisi_del_Trecento_recessione_e_Innovazione_-_ExtraBarbero_Grattacielo_Intesa_Sanpaolo_2019_0027.wav"
REFERENCE_TEXT = "ha avuto i suoi alti e bassi."
TEXT = "<|speaker:0|>Questo è un test di generazione audio."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = torch.bfloat16


def build_mock_trainer_and_module(tmpdir: str):
    """Build a real model wrapped in minimal mocks for Trainer/LightningModule."""
    from fish_speech.models.text2semantic.llama import BaseTransformer

    logger.info("Loading model...")
    model = BaseTransformer.from_pretrained(
        CHECKPOINT_PATH, load_weights=True, max_length=4096
    )
    model = model.to(device=DEVICE, dtype=PRECISION)

    # Mock LightningModule
    pl_module = MagicMock()
    pl_module.model = model
    pl_module.device = torch.device(DEVICE)

    # Mock Trainer
    trainer = MagicMock()
    trainer.global_step = 100
    trainer.global_rank = 0
    trainer.log_dir = tmpdir
    trainer.loggers = []
    # Mock optimizer with empty state (nothing to offload)
    trainer.optimizers = []

    return trainer, pl_module


def main():
    tmpdir = tempfile.mkdtemp(prefix="audio_callback_test_")
    logger.info(f"Output dir: {tmpdir}")

    try:
        trainer, pl_module = build_mock_trainer_and_module(tmpdir)

        from fish_speech.callbacks.audio_sample import AudioSampleCallback

        callback = AudioSampleCallback(
            text=TEXT,
            codec_checkpoint_path=CODEC_PATH,
            reference_audio_path=REFERENCE_AUDIO,
            reference_text=REFERENCE_TEXT,
            output_dir=tmpdir,
        )

        logger.info("Running _generate_sample...")
        callback._generate_sample(trainer, pl_module)

        # Check output
        wav_files = list(Path(tmpdir).glob("*.wav"))
        if wav_files:
            for f in wav_files:
                size_kb = f.stat().st_size / 1024
                logger.info(f"SUCCESS: {f.name} ({size_kb:.1f} KB)")
        else:
            logger.error("FAIL: No wav files generated")
            return 1

        # Verify model is back in train mode and caches are gone
        model = pl_module.model
        assert model.training, "Model should be back in training mode"
        has_cache = any(
            "kv_cache" in layer.attention._modules for layer in model.layers
        )
        assert not has_cache, "KV caches should be torn down"
        logger.info("All assertions passed.")
        return 0

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
