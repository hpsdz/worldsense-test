# Sources

- omniscope.py: functions extracted with Python AST (formatting/comments changed)
  from https://github.com/MAC-AutoML/OmniScope at
  d8fb72ff2e45876646fc9fab5fa50234eaf5a6ed: tools/modeling_qwen2_5_omni.py
  (token_pruning and audio merging), tools/qwenomni_prune_inference.py (audio
  allocation), eval_worldSense.py (neighbor boost). Adapter and CLIP preparation
  are in compression/omniscope.py. Apache 2.0 license retained in
  docs/upstream/OmniScope-LICENSE. The upstream full model/processor is not imported.

- omnizip.py: copied from OmniZip-main/omnizip/omnizip_core.py in this workspace.
- omniselect.py: copied from OmniSelect/OmniSelect/OmniSelect.py, including the
  explicitly documented local sparse-group and boundary fixes made for the demo.

These algorithms belong to the respective upstream authors. Keep upstream
repositories, attribution and applicable licensing when redistributing. See the
respective repository README/license files. No Qwen modeling implementation is
copied into this package. Shared adapters are maintained separately.

- omnipack/pre_llm_units.py and inner_llm_units.py: unmodified copies from
  https://github.com/RowanSu/OmniPack at aecfa53c6c041eae35a0eb3bc287e2d6671d994c.
  compression/omnipack.py adapts the forward patches and VisionZip encoder
  importance formulas to the shared SDPA model; physical cache positions are
  compacted while semantic RoPE positions are preserved. See README.md and
  docs/history/restructure-20261108.md (historical snapshot).
  The cloned OmniPack root has no LICENSE file; attribution is not a substitute
  for a redistribution license. Upstream README is retained in docs/upstream.
- OmniSelect upstream: https://github.com/Yangmrl-nlp/OmniSelect.
  Original code and local changes are archived in docs/legacy/legacy-sources.zip.
  OmniZip license and all upstream READMEs are retained in docs/upstream;
  AudioCLIP license is retained in vendor/audioclip/LICENSE.md.
