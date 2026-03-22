# Fine-tuning con LoRA (s2-pro su RTX 5090)

Questa guida documenta il processo completo per fare fine-tuning con LoRA sul modello `s2-pro`,
incluse tutte le correzioni necessarie per farlo funzionare.

## Fix applicati al codice

### 1. `fish_speech/models/text2semantic/lora.py`
Il modello `s2-pro` usa `tie_word_embeddings=True`, quindi non crea il layer `output`.
Il codice originale crashava cercando di wrapparlo con LoRA.
Fix: controllare con `hasattr` prima di aggiungerlo.

### 2. `fish_speech/configs/text2semantic_finetune.yaml`
- `pretrained_ckpt_path`: cambiato da `checkpoints/openaudio-s1-mini` a `checkpoints/s2-pro`
- `tokenizer.model_path`: cambiato da `${pretrained_ckpt_path}/tokenizer.tiktoken` a `${pretrained_ckpt_path}` (s2-pro usa tokenizer standard HuggingFace, non il file .tiktoken)
- `batch_size`: ridotto da 4 a 1 (il modello da 4.6B occupa ~20GB a max_length=4096 su RTX 5090 32GB)
- `accumulate_grad_batches`: aumentato da 1 a 4
- `max_length`: 4096 (i clip di 30s producono ~2700 token, abbondantemente sotto il limite)
- `max_steps`: 5000
- `val_check_interval`: 1000 (salva checkpoint ogni 1000 step → 5 checkpoint totali)

## Preparazione del dataset

Il dataset deve essere strutturato con una cartella per speaker, con file `.wav` e `.lab` accoppiati:

```
data/
└── barbero/
    ├── clip_0000.wav
    ├── clip_0000.lab
    ├── clip_0001.wav
    ├── clip_0001.lab
    └── ...
```

### Token rate del codec

Il codec DAC opera a 44100 Hz con hop_length=512 (strides 2×4×8×8):
- **86 token/secondo** di audio
- Clip da 20s → ~1720 token audio + testo ≈ 1850 token totali
- Clip da 30s → ~2580 token audio + testo ≈ 2700 token totali
- max_length=4096 copre clip fino a ~45 secondi senza troncamenti

---

## Dataset di test (32 clip — `data/barbero_test`)

### Step 1 — Estrazione token VQ

```bash
uv run python tools/vqgan/extract_vq.py data/barbero_test --num-workers 1 --batch-size 16 --config-name "modded_dac_vq" --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

### Step 2 — Creazione protobuf

```bash
uv run python tools/llama/build_dataset.py --input "data/barbero_test" --output "data/protos_test" --text-extension .lab --num-workers 4
```

### Training

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python fish_speech/train.py --config-name text2semantic_finetune project=barbero_test +lora@model.model.lora_config=r_8_alpha_16 train_dataset.proto_files=[data/protos_test] val_dataset.proto_files=[data/protos_test]
```

---

## Dataset completo (`data/barbero`)

### Step 1 — Estrazione token VQ

```bash
uv run python tools/vqgan/extract_vq.py data/barbero --num-workers 1 --batch-size 16 --config-name "modded_dac_vq" --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

### Step 2 — Creazione protobuf

```bash
uv run python tools/llama/build_dataset.py --input "data/barbero" --output "data/protos" --text-extension .lab --num-workers 4
```

### Training

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python fish_speech/train.py --config-name text2semantic_finetune project=barbero_lora +lora@model.model.lora_config=r_8_alpha_16
```

---

## Merge dei pesi LoRA

Dopo il training, unire i pesi LoRA al modello base. Usare il checkpoint più basso che suona bene, non necessariamente l'ultimo.

```bash
uv run python tools/llama/merge_lora.py --lora-config r_8_alpha_16 --base-weight checkpoints/s2-pro --lora-weight results/barbero_lora/checkpoints/step_000001000.ckpt --output checkpoints/s2-pro-barbero-lora/
```

---

## Note

- Il modello s2-pro è RL-trained: il fine-tuning può degradare la qualità generale. Usare pochi step e il checkpoint più conservativo.
- Il LoRA impara i pattern del parlato, non il timbro. Usare sempre un reference audio per il timbro in inferenza.
- Con RTX 5090 (32GB), `batch_size=1` usa ~20GB a max_length=4096. Non c'è margine per batch_size=2.
- Monitorare VRAM con `watch -n1 nvidia-smi` durante le prime iterazioni.
