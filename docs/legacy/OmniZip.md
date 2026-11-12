# Local demo compatibility

Use the `omnizip` Conda environment (PyTorch 2.6.0+cu124, Transformers 4.52.3).
From the parent project folder:

```powershell
conda activate omnizip
python OmniZip-main/demo.py
python OmniZip-main/demo.py --omnizip
```

The default video remains this repository's `assets/example.mp4` and the prompt
remains `Describe this video.`. The default model is the existing local
`D:/Models/Qwen2.5-Omni-3B`; override with `--model`.

Changes for this 8 GB Windows GPU:
- Resolve the default video relative to demo.py, independent of the working directory.
- Use SDPA instead of requiring the unavailable FlashAttention2 package.
- Disable speech generation at model construction; automatically place weights with
  5 GiB GPU / 24 GiB CPU budgets to reserve inference memory.
- Limit inputs to 16 frames and 100352 pixels per frame; expose `--max-frames`,
  `--max-pixels`, and `--max-new-tokens` (default 128).
- Forward actual sampled FPS to the processor to preserve audiovisual timing.
- Use the parent project's `vendor/qwen-omni-utils/src` snapshot, including its
  PyAV audio decoder, rather than another checkout's editable installation.
- Use greedy text generation with an explicit output limit.
- Fix SDPA audio importance scoring: Q/K are already [heads, time, dim], so do not
  transpose them again. The FlashAttention branch and compression core are unchanged.
- Print the actual sequence token count before/after OmniZip compression.

These are low-memory demo settings, not a reproduction of paper benchmark results.
The parent project's vendor directory must accompany this checkout when migrating.
Baseline mode imports the installed Transformers model; `--omnizip` imports
`omnizip/modeling_qwen2_5_omni.py`.
