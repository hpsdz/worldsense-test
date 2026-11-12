# Local single-video demo

Run from the parent qwen directory in PowerShell:

```powershell
& C:/Users/wyz/.conda/envs/omnizip/python.exe -B ./OmniSelect/demo.py
# Shorter output, or a custom video:
& C:/Users/wyz/.conda/envs/omnizip/python.exe -B ./OmniSelect/demo.py --max-new-tokens 32
& C:/Users/wyz/.conda/envs/omnizip/python.exe -B ./OmniSelect/demo.py --video D:/Datasets/WorldSense/videos/FnOIAada.mp4
```

Default input is the sibling OmniZip-main/assets/example.mp4. Default model is
D:/Models/Qwen2.5-Omni-3B. Default prompt: Describe this video.
Uses 16 frames, max_pixels=100352, requested 2 FPS (actual sampled FPS forwarded
to the processor), full audio, SDPA, no weight quantization, 64 output tokens,
GPU budget 5 GiB and CPU budget 24 GiB. The CPU offload makes timings unsuitable
for comparison with published GPU benchmarks. AudioCLIP is released before Qwen
loads. Results overwrite results/demo.json unless --output is specified.

## Environment and assets

Uses the existing omnizip environment: Python 3.10, torch 2.6.0+cu124,
torchvision 0.21.0+cu124, transformers 4.52.3. Only termcolor==3.3.0 was newly
installed during setup; nltk and ftfy were already available. requirements-demo.txt
lists the extra inference dependencies relative to the parent environment.
Do not install the original full requirements.txt blindly: it includes Linux
wheel URLs and absolute paths from the author's machine.

Required locally downloaded assets (ignored by Git):

- AudioCLIP/assets/AudioCLIP-Full-Training.pt (537302068 bytes):
  https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
- AudioCLIP/assets/bpe_simple_vocab_16e6.txt.gz:
  https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
- assets/nltk_data/tokenizers/punkt_tab and assets/nltk_data/taggers/averaged_perceptron_tagger_eng:
  extracted from the corresponding zip packages at
  https://github.com/nltk/nltk_data/tree/gh-pages/packages

Preprocessing imports the parent's vendor/qwen-omni-utils/src snapshot, including
PyAV audio decoding; keep that directory when moving the demo to another machine.
The input must contain both video and audio. This is single-sample inference.

## Changes

- Added demo.py with CLI paths, actual AudioCLIP query relevance, automatic modality
  choice, Qwen generation and JSON reporting of per-modality token counts.
- Corrected AudioCLIP text input nesting: each keyword is a list item containing
  a word, rather than a string that AudioCLIP joins character by character.
- Fixed the OmniSelect model's SDPA audio Q/K axes and returned a one-dimensional
  token importance vector using the same chunked reduction as its FlashAttention
  branch. Added compression statistics, a relative algorithm import and argument
  documentation. The FlashAttention path was not changed or validated here.
- Added an explicit-group algorithm path used by this demo. Scores follow actual
  processor token runs, including runs splitting a visual temporal patch. This
  removes fixed four-frame assumptions and off-by-one/empty-group problems in
  the original iteration. Audio scores are weighted by overlap with audio-token
  groups; video scores come from pairs of sampled frames. Original modality
  selection, sigmoid budget allocation, audio top-k and video bottom-k ranking
  rules are retained. Original evaluation scripts retain their legacy group path.
- Added finite-score/length checks and a single-group standard-deviation guard;
  removed an unused matplotlib import.
- Made AudioCLIP's training entry points lazy imports so inference does not
  require ignite/visdom. Added asset/output ignore rules.

These are local compatibility adaptations, not an exact reproduction of the
paper's evaluation configuration. The user's existing scripts/infer_worldsense.sh
changes were preserved. Parent WorldSense and OmniZip sources were not edited.

The reported generation time includes Qwen encoding, pruning and decoding but
excludes AudioCLIP, preprocessing and model loading; total time includes those.
The demo asserts token reduction and unchanged query-token count. Query includes
text and special tokens, consistent with the parent test project's convention.

## Verified run

Command: `demo.py --max-new-tokens 32`, using all other defaults, exited 0.
AudioCLIP chose video-first (audio score 3.461, video score 7.392).
Video tokens: 1152 -> 462; audio: 795 -> 477; query: 56 -> 56;
total: 2003 -> 995. Output described a woman presenting a charger stand and
accessories. It reached the 32-token output limit, so the description is truncated.
Generation took 40.34 seconds; total including loading was 54.28 seconds.
Synthetic smoke checks also exercised audio-first and uniform branches; those
two branches have not been validated with end-to-end video generation here.

Temporary download_parts and partial-download files remain ignored locally:
automatic command policy rejected their cleanup. They are not needed for inference.
