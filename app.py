"""Run local Qwen2.5-Omni on WorldSense multiple-choice questions."""
import argparse
import gc
import inspect
import json
import math
import re
import sys
import time
import statistics
from datetime import datetime
from pathlib import Path

# Pin all strategies to one preprocessing snapshot.
sys.path.insert(0, str(Path(__file__).parent / "vendor/qwen-omni-utils/src"))

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def load_config(path):
    """Load validated JSON defaults; resolve file paths relative to the config."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")
    integers = {"limit", "offset", "max_frames", "max_pixels", "max_new_tokens"}
    booleans = {"short_only", "use_audio_in_video", "do_sample", "load_in_4bit"}
    strings = {"dataset", "model", "gpu_memory", "cpu_memory", "attn_implementation", "compression_method"}
    nullable_strings = {"video_id", "output"}
    lists = {"tasks", "task_domain", "task_type"}
    allowed = integers | booleans | strings | nullable_strings | lists | {"fps", "max_duration", "compression_params"}
    for key, value in data.items():
        if key not in allowed:
            raise ValueError(f"Unknown config key: {key}")
        valid = True
        if key == "compression_params":
            valid = isinstance(value, dict)
        elif key in integers:
            valid = type(value) is int
        elif key in booleans:
            valid = type(value) is bool
        elif key in strings:
            valid = isinstance(value, str) and bool(value.strip())
        elif key in nullable_strings:
            valid = value is None or (isinstance(value, str) and bool(value.strip()))
        elif key in lists:
            valid = value is None or (isinstance(value, list) and bool(value)
                                      and all(isinstance(v, str) and v.strip() for v in value))
        else:
            valid = (key == "max_duration" and value is None) or (
                type(value) in (int, float) and math.isfinite(value) and value > 0)
        if not valid:
            raise ValueError(f"Invalid value for {key}: {value!r}")
    if "attn_implementation" in data and data["attn_implementation"] not in {"sdpa", "eager", "flash_attention_2"}:
        raise ValueError("Unsupported attn_implementation")
    for key in ("gpu_memory", "cpu_memory"):
        if key in data and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:GiB|MiB|GB|MB)", data[key]):
            raise ValueError(f"Invalid memory budget: {key}")
    for key in ("dataset", "model", "output"):
        if data.get(key) is not None:
            data[key] = str((path.parent / data[key]).resolve())
    for key in ("audioclip", "clip_model", "nltk_data"):
        if "compression_params" in data and key in data["compression_params"]:
            data["compression_params"][key] = str((path.parent / data["compression_params"][key]).resolve())
    return data


def parse_duration_seconds(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", str(value).strip())
    if not match:
        raise ValueError(f"Invalid video duration: {value!r}")
    return float(match.group(1))


def load_dataset(root):
    root = Path(root)
    with (root / "worldsense_qa.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    samples, missing = [], []
    for key, video in data.items():
        video_id = video["video_id"]
        if key != video_id:
            raise ValueError(f"Video ID mismatch: {key} != {video_id}")
        video_path = root / "videos" / f"{video_id}.mp4"
        duration_seconds = parse_duration_seconds(video["video_duration"])
        if not video_path.is_file():
            missing.append(video_id)
        task_keys = sorted((k for k in video if re.fullmatch(r"task\d+", k)),
                           key=lambda k: int(k[4:]))
        for task_id in task_keys:
            task = video[task_id]
            candidates = task["candidates"]
            labels = [chr(65 + i) for i in range(len(candidates))]
            if (not task["question"] or not candidates or task["answer"] not in labels
                    or any(not option.startswith(f"{label}.")
                           for label, option in zip(labels, candidates))):
                raise ValueError(f"Invalid annotation: {video_id}/{task_id}")
            samples.append({
                "video_id": video_id, "task_id": task_id, "video_path": str(video_path),
                "domain": video.get("domain"), "task_domain": task["task_domain"],
                "task_type": task["task_type"],
                "duration_seconds": duration_seconds,
                "question": task["question"], "candidates": candidates, "answer": task["answer"],
            })
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} videos: {missing[:10]}")
    if not samples:
        raise ValueError("No WorldSense questions found.")
    return samples, len(data)


def select_samples(samples, max_duration=None):
    """Select complete annotated questions from videos up to max_duration seconds."""
    if max_duration is None:
        return list(samples)
    if max_duration <= 0:
        raise ValueError("max_duration must be positive")
    return [sample for sample in samples if sample["duration_seconds"] <= max_duration]


def select_tasks(samples, task_ids=None):
    """Select only the requested task IDs from an already filtered video set."""
    if task_ids is None:
        return list(samples)
    requested = set(task_ids)
    if not requested or any(not re.fullmatch(r"task\d+", task_id) for task_id in requested):
        raise ValueError("task_ids must contain values such as task0 or task2")
    return [sample for sample in samples if sample["task_id"] in requested]


def select_metadata(samples, domains=None, task_types=None):
    """Filter questions by task domain and/or task type."""
    domain_set = set(domains) if domains else None
    type_set = set(task_types) if task_types else None
    if domain_set and any(not isinstance(value, str) or not value.strip() for value in domain_set):
        raise ValueError("domains must contain non-empty task domain names")
    if type_set and any(not isinstance(value, str) or not value.strip() for value in type_set):
        raise ValueError("task_types must contain non-empty task type names")
    return [sample for sample in samples
            if (domain_set is None or sample["task_domain"] in domain_set)
            and (type_set is None or sample["task_type"] in type_set)]


def accuracy_by(results, field):
    """Return per-category counts and accuracy using all processed rows."""
    grouped = {}
    for result in results:
        key = result.get(field)
        if key is None:
            continue
        group = grouped.setdefault(key, {"correct": 0, "total": 0, "accuracy": None})
        group["total"] += 1
        group["correct"] += int(bool(result.get("correct")))
    for group in grouped.values():
        group["accuracy"] = group["correct"] / group["total"] if group["total"] else None
    return dict(sorted(grouped.items()))


def build_conversation(sample, max_frames, max_pixels, fps=2.0):
    # Gold answers and human-written captions must never enter the model prompt.
    prompt = (sample["question"] + "\n" + "\n".join(sample["candidates"])
              + "\nSelect the best answer based on the video and its audio. "
              "Reply with only the option letter (for example, A).")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": sample["video_path"],
             "max_frames": max_frames, "max_pixels": max_pixels, "fps": fps},
            {"type": "text", "text": prompt},
        ]},
    ]


def parse_answer(response, candidates):
    # Reject ambiguous prose instead of silently selecting the first capital letter.
    match = re.fullmatch(r"\s*(?:Answer\s*:\s*)?\(?([A-Z])\)?[.!]?\s*",
                         response, flags=re.IGNORECASE)
    if match and ord(match[1].upper()) - ord("A") < len(candidates):
        return match[1].upper()
    for candidate in candidates:
        if response.strip().casefold() == candidate.strip().casefold():
            return candidate[0]
    return None


def count_input_tokens(input_ids, attention_mask, video_token_id, audio_token_id):
    """Count one processed input sequence, excluding padding and generated output."""
    if len(input_ids) != len(attention_mask):
        raise ValueError("Input IDs and attention mask lengths differ")
    active = [token for token, mask in zip(input_ids, attention_mask) if mask]
    video = active.count(video_token_id)
    audio = active.count(audio_token_id)
    return {"video_tokens": video, "audio_tokens": audio,
            "query_tokens": len(active) - video - audio, "total_tokens": len(active)}


def summarize_input_tokens(results):
    measured = [row["input_tokens"] for row in results if row.get("input_tokens") is not None]
    return {
        "counted_inputs": len(measured),
        "uncounted_inputs": len(results) - len(measured),
        "averages": {key: sum(row[key] for row in measured) / len(measured) if measured else None
                     for key in ("video_tokens", "audio_tokens", "query_tokens", "total_tokens")},
        "unit": "per video-task input; repeated videos count once per task",
    }


def summarize_compression(results):
    measured = [r["compression_tokens"] for r in results if r.get("compression_tokens")]
    stages = ["before", "after"]
    if measured and all("after_stage1" in r["total_tokens"] for r in measured):
        stages.insert(1, "after_stage1")
    return {"measured":len(measured), "averages": {
        key: {stage:sum(r[key][stage] for r in measured)/len(measured) for stage in stages}
        for key in ("video_tokens","audio_tokens","query_tokens","total_tokens")
    } if measured else {}}


class RunTiming:
    """Wall-clock snapshots; final report writing is outside the measured interval."""
    def __init__(self):
        self.started = time.perf_counter()
        self.evaluation_started = None
        self.model_load_seconds = None

    def snapshot(self, finalized=False):
        now = time.perf_counter()
        return {
            "run_wall_seconds": now - self.started,
            "model_load_seconds": self.model_load_seconds,
            "evaluation_wall_seconds": (now - self.evaluation_started
                                        if self.evaluation_started is not None else None),
            "finalized": finalized,
        }


def summarize_timing(results, timing=None):
    fields = ("elapsed_seconds", "strategy_prepare_seconds", "generation_seconds")
    def aggregate(rows):
        summary = {}
        for field in fields:
            values = [r[field] for r in rows if type(r.get(field)) in (int, float)
                      and math.isfinite(r[field]) and r[field] >= 0]
            summary[field] = {
                "count": len(values), "missing_count": len(rows) - len(values),
                "total_seconds": sum(values) if values else None,
                "mean_seconds": statistics.mean(values) if values else None,
                "median_seconds": statistics.median(values) if values else None,
            }
        return summary
    return {
        "unit": "seconds",
        "global": timing or {"run_wall_seconds": None, "model_load_seconds": None,
                             "evaluation_wall_seconds": None, "finalized": False},
        "all_processed": aggregate(results),
        "completed": aggregate([r for r in results if r.get("status") in ("ok", "invalid_answer")]),
        "scope": "Run: main entry through final cleanup; evaluation: initial checkpoint through final cleanup, including per-question saves. Both exclude final report write/printing and process teardown. Completed excludes inference errors, includes invalid answers. Missing phase times are excluded, not zero-filled.",
    }


def save_results(path, config, results, selected_count, timing=None):
    correct = sum(r["correct"] for r in results)
    report = {
        "config": config,
        "summary": {
            "selected": selected_count, "processed": len(results),
            "completed": sum(r["status"] != "error" for r in results),
            "errors": sum(r["status"] == "error" for r in results),
            "invalid_answers": sum(r["status"] == "invalid_answer" for r in results),
            "correct": correct,
            "accuracy": correct / len(results) if results else None,
            "accuracy_denominator": "all processed questions, including errors and invalid answers",
            "accuracy_by_task_domain": accuracy_by(results, "task_domain"),
            "accuracy_by_task_type": accuracy_by(results, "task_type"),
            "input_tokens": summarize_input_tokens(results),
            "compression": summarize_compression(results),
            "timing": summarize_timing(results, timing),
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return report["summary"]


def infer(sample, args, model, processor, torch, process_mm_info, result=None):
    conversation = build_conversation(sample, args.max_frames, args.max_pixels, args.fps)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos, video_kwargs = process_mm_info(
        conversation, use_audio_in_video=args.use_audio_in_video, return_video_kwargs=True,
    )
    # The installed processor expects scalar fps for one video. Forward actual
    # sampled fps to keep sparse frames synchronized with the full audio track.
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        fps=video_kwargs["fps"][0], return_tensors="pt", padding=True,
        use_audio_in_video=args.use_audio_in_video,
    )
    token_counts = count_input_tokens(
        inputs.input_ids[0].tolist(), inputs.attention_mask[0].tolist(),
        processor.tokenizer.convert_tokens_to_ids(processor.video_token),
        processor.tokenizer.convert_tokens_to_ids(processor.audio_token),
    )
    # Save counts before generation so OOM/decoding failures retain measured inputs.
    if result is not None:
        result["input_tokens"] = token_counts
    print(f"Input tokens: {json.dumps(token_counts)}", flush=True)
    strategy = model.thinker.compression_strategy
    model.thinker.compression_context = None
    model.thinker.last_compression_stats = None
    preparation_start = time.perf_counter()
    prepare_kwargs = {"tokenizer": processor.tokenizer, "candidates": sample["candidates"]} if args.compression_method == "omniscope" else {}
    model.thinker.compression_context = strategy.prepare(
        frames=videos[0], audio=audios[0] if audios else None, question=sample["question"],
        inputs=inputs, audio_id=model.thinker.config.audio_token_id, video_id=model.thinker.config.video_token_id,
        **prepare_kwargs)
    preparation_seconds = time.perf_counter()-preparation_start
    if result is not None:
        result["strategy_prepare_seconds"] = preparation_seconds
    inputs = inputs.to(model.device).to(model.dtype)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    generation_start = time.perf_counter()
    with torch.inference_mode():
        text_ids = model.generate(
            **inputs, use_audio_in_video=args.use_audio_in_video, return_audio=False,
            thinker_do_sample=args.do_sample, thinker_max_new_tokens=args.max_new_tokens,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter()-generation_start
    if result is not None:
        stats = model.thinker.last_compression_stats
        if args.compression_method != "baseline" and stats is None:
            raise RuntimeError("Selected compression strategy did not execute")
        result["compression_tokens"] = stats or {k:{"before":v,"after":v} for k,v in token_counts.items()}
        result["strategy_prepare_seconds"] = preparation_seconds
        result["generation_seconds"] = generation_seconds
        result["generated_tokens"] = text_ids.shape[1]-inputs.input_ids.shape[1]
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        result["sampled_frames"] = int(videos[0].shape[0])
        result["sampled_fps"] = video_kwargs["fps"][0]
        context = model.thinker.compression_context
        if context and "video_first" in context:
            result["modality_priority"] = context["video_first"]
    return processor.batch_decode(
        text_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def build_quantization_config(enabled):
    """Optional CUDA NF4 weight quantization; never silently fall back."""
    if not enabled:
        return None
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit mode requires a CUDA GPU in this application.")
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("4-bit mode requires compatible bitsandbytes and transformers in the active environment.") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, help="Question count; 0 means all remaining questions")
    parser.add_argument("--offset", type=int, help="Skip questions in annotation order after filtering")
    parser.add_argument("--video-id", help="Only evaluate questions for this video")
    parser.add_argument("--tasks", nargs="+", metavar="TASK_ID",
                        help="Only evaluate these tasks for --video-id, for example: task0 task2")
    parser.add_argument("--task-domain", nargs="+", metavar="DOMAIN",
                        help="Only evaluate these task domains, for example: Recognition Reasoning")
    parser.add_argument("--task-type", nargs="+", metavar="TYPE",
                        help="Only evaluate these task types; quote names containing spaces")
    parser.add_argument("--short-only", action=argparse.BooleanOptionalAction,
                        help="Only evaluate videos whose annotated duration is <= 30 seconds")
    parser.add_argument("--max-duration", type=float,
                        help="Only evaluate videos up to this annotated duration in seconds")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output", type=Path, help="JSON result path (default: timestamped file in results/)")
    parser.add_argument("--check-data", action="store_true", help="Validate all annotations and paths without loading the model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--use-audio-in-video", action=argparse.BooleanOptionalAction)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction,
                        help="Enable NF4 weight quantization (CUDA and bitsandbytes required)")
    parser.add_argument("--gpu-memory")
    parser.add_argument("--cpu-memory")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--compression-method", default="baseline")
    parser.add_argument("--compression-params", type=json.loads, default={})
    # Read config before parsing the full CLI; explicit CLI arguments win.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_path = probe.parse_known_args(argv)[0].config.resolve()
    try:
        defaults = load_config(DEFAULT_CONFIG)
        if config_path != DEFAULT_CONFIG.resolve():
            defaults.update(load_config(config_path))
        parser.set_defaults(**defaults)
    except (OSError, ValueError) as exc:
        parser.error(f"Invalid config: {exc}")
    args = parser.parse_args(argv)
    from compression import create_strategy
    try:
        create_strategy(args.compression_method, args.compression_params)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.config = config_path
    # Config paths are relative to that config; CLI paths retain shell semantics.
    for key in ("dataset", "model", "output"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, Path(value))
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps must be finite and positive")
    if args.max_duration is not None and not math.isfinite(args.max_duration):
        parser.error("max-duration must be finite")
    if (args.limit < 0 or args.offset < 0 or args.max_frames < 2
            or args.max_frames % 2 or args.max_pixels < 128 * 28 * 28
            or args.max_new_tokens < 1 or (args.max_duration is not None and args.max_duration <= 0)):
        parser.error("Use nonnegative limit/offset, even max-frames >= 2, max-pixels >= 100352, and positive max-new-tokens.")
    if args.tasks and not args.video_id:
        parser.error("--tasks requires --video-id so tasks cannot be mixed across videos.")
    return parser, args


def main():
    run_timing = RunTiming()
    parser, args = parse_args()
    samples, video_count = load_dataset(args.dataset)
    print(f"WorldSense: {video_count} videos, {len(samples)} questions; all video paths and annotations validated.", flush=True)
    max_duration = 30.0 if args.short_only else args.max_duration
    if args.short_only and args.max_duration is not None:
        parser.error("Use either --short-only or --max-duration, not both.")
    if args.video_id:
        samples = [s for s in samples if s["video_id"] == args.video_id]
        if not samples:
            parser.error(f"Unknown video id: {args.video_id}")
    try:
        samples = select_tasks(samples, args.tasks)
        samples = select_metadata(samples, args.task_domain, args.task_type)
    except ValueError as exc:
        parser.error(str(exc))
    samples = select_samples(samples, max_duration)
    selected_before_limit = len(samples)
    selected_video_count = len({s["video_id"] for s in samples})
    if not samples:
        parser.error("No questions match the requested video-id, task, domain, type, or duration filter.")
    print(f"Selected {selected_video_count} videos / {selected_before_limit} questions"
          f" (duration <= {max_duration:g}s)." if max_duration is not None
          else f"Selected {selected_video_count} videos / {selected_before_limit} questions.", flush=True)
    if args.check_data:
        return 0
    samples = samples[args.offset:]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        parser.error("No questions match the requested video-id/offset.")
    output = args.output or Path(__file__).parent / "results" / f"worldsense_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output = output.resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new --output path: {output}")
    if output.suffix.lower() != ".json":
        parser.error("--output must use a .json extension.")
    try:
        import torch
        import transformers
        from models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        from transformers import Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}. Run: conda run --no-capture-output -n omnizip python app.py") from exc

    from compression import create_strategy
    strategy = create_strategy(args.compression_method, args.compression_params)
    from compression.resources import resource_path
    for resource in ("audioclip", "clip_model", "nltk_data"):
        if resource in strategy.params:
            strategy.params[resource] = resource_path(resource, strategy.params[resource])
    if args.compression_method != "baseline" and (not args.use_audio_in_video or args.attn_implementation != "sdpa"):
        parser.error("Compressed inference currently requires audio and SDPA")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    print("Loading model...", flush=True)
    model_source = str(Path(inspect.getfile(Qwen2_5OmniForConditionalGeneration)).resolve())
    print(f"Model implementation: {model_source}", flush=True)
    quantization_config = build_quantization_config(args.load_in_4bit)
    quantization_kwargs = {"quantization_config": quantization_config} if quantization_config else {}
    model_load_start = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype="auto", device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory} if torch.cuda.is_available() else None,
        attn_implementation=args.attn_implementation, enable_audio_output=False, local_files_only=True,
        **quantization_kwargs,
    ).eval()
    model.thinker.compression_strategy = strategy
    model.thinker.compression_method = args.compression_method
    model.thinker.compression_context = None
    if args.load_in_4bit and not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("4-bit was requested but model did not load in 4-bit mode.")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(args.model), local_files_only=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_timing.model_load_seconds = time.perf_counter() - model_load_start
    config = {
        "compression": {"method":args.compression_method, "params":strategy.params,
                        "implementation":"local-adapted-v1", "cache":"compact DynamicCache with original RoPE positions"},
        "preprocessing_source": inspect.getfile(process_mm_info),
        "quantization": quantization_config.to_dict() if quantization_config else None,
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "token_count_definition": "Count non-padding processed input_ids. Video/audio count expanded modality placeholders; query is all remaining input tokens, including system prompt, question, options, instructions and template/modality boundary tokens. Excludes generated output.",
        "model_implementation": model_source,
        "effective_parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset": str(args.dataset.resolve()), "model": str(args.model), "video_count": video_count,
        "video_id_filter": args.video_id, "offset": args.offset, "limit": args.limit,
        "task_filter": sorted(set(args.tasks)) if args.tasks else None,
        "task_domain_filter": sorted(set(args.task_domain)) if args.task_domain else None,
        "task_type_filter": sorted(set(args.task_type)) if args.task_type else None,
        "max_duration_seconds": max_duration,
        "max_frames": args.max_frames, "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens, "use_audio_in_video": args.use_audio_in_video,
        "do_sample": args.do_sample, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "evaluation": "local low-memory sample evaluation, not the official benchmark protocol",
    }
    results = []
    run_timing.evaluation_started = time.perf_counter()
    save_results(output, config, results, len(samples), run_timing.snapshot())
    for index, sample in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] {sample['video_id']}/{sample['task_id']}: {sample['question']}", flush=True)
        start = time.perf_counter()
        result = dict(sample, response=None, prediction=None, correct=False, input_tokens=None)
        try:
            response = infer(sample, args, model, processor, torch, process_mm_info, result=result)
            prediction = parse_answer(response, sample["candidates"])
            result.update(response=response, prediction=prediction,
                          correct=prediction == sample["answer"],
                          status="ok" if prediction else "invalid_answer")
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(result["error"], flush=True)
        result["elapsed_seconds"] = time.perf_counter() - start
        results.append(result)
        summary = save_results(output, config, results, len(samples), run_timing.snapshot())
        print(f"Prediction={result['prediction']}, gold={sample['answer']}, correct={result['correct']}, status={result['status']}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    summary = save_results(output, config, results, len(samples), run_timing.snapshot(finalized=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {output}", flush=True)
    return 1 if summary["errors"] or summary["invalid_answers"] else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
