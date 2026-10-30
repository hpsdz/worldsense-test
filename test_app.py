import json
import tempfile
import unittest
from pathlib import Path
from app import load_config, parse_args
from app import build_quantization_config
from app import count_input_tokens, summarize_input_tokens

from app import (accuracy_by, build_conversation, load_dataset, parse_answer,
                 save_results, select_metadata, select_samples, select_tasks)


class WorldSenseTests(unittest.TestCase):
    def test_quantization_toggle_and_cli_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"load_in_4bit": True}), encoding="utf-8")
            self.assertTrue(parse_args(["--config", str(path)])[1].load_in_4bit)
            self.assertFalse(parse_args(["--config", str(path), "--no-load-in-4bit"])[1].load_in_4bit)
            path.write_text(json.dumps({"load_in_4bit": "false"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
        self.assertIsNone(build_quantization_config(False))

    def test_token_counts_exclude_padding(self):
        counts = count_input_tokens([1, 9, 9, 8, 3, 9], [1, 1, 1, 1, 1, 0], 9, 8)
        self.assertEqual(counts, {"video_tokens": 2, "audio_tokens": 1,
                                  "query_tokens": 2, "total_tokens": 5})
        self.assertEqual(count_input_tokens([1, 9, 3], [1, 1, 1], 9, 8)["audio_tokens"], 0)

    def test_token_means_include_measured_failures_not_missing_counts(self):
        a = count_input_tokens([9, 8, 1], [1, 1, 1], 9, 8)
        b = count_input_tokens([9, 9, 1, 2], [1, 1, 1, 1], 9, 8)
        summary = summarize_input_tokens([
            {"input_tokens": a, "status": "ok"},
            {"input_tokens": b, "status": "error"},
            {"input_tokens": None, "status": "error"}])
        self.assertEqual(summary["counted_inputs"], 2)
        self.assertEqual(summary["uncounted_inputs"], 1)
        self.assertEqual(summary["averages"]["video_tokens"], 1.5)
        self.assertEqual(summary["averages"]["audio_tokens"], 0.5)
        self.assertEqual(summary["averages"]["query_tokens"], 1.5)
        self.assertIsNone(summarize_input_tokens([])["averages"]["query_tokens"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "videos").mkdir()
        (self.root / "videos" / "sample.mp4").touch()
        self.data = {"sample": {
            "video_id": "sample", "video_duration": "20s", "domain": "Music",
            "video_caption": "SECRET HUMAN CAPTION",
            "task0": {"task_domain": "Recognition", "question": "Which instrument is playing?", "answer": "B",
                      "candidates": ["A. Piano", "B. Guitar"], "task_type": "Recognition"},
        }}
        self.write_data()

    def write_data(self):
        (self.root / "worldsense_qa.json").write_text(json.dumps(self.data), encoding="utf-8")

    def test_config_overrides_and_relative_paths(self):
        config = self.root / "experiment.json"
        config.write_text(json.dumps({"dataset": "data", "max_frames": 24,
                                      "limit": 0, "short_only": True,
                                      "use_audio_in_video": False,
                                      "task_type": ["Audio Recognition"]}), encoding="utf-8")
        _, args = parse_args(["--config", str(config), "--max-frames", "16", "--no-short-only"])
        self.assertEqual(args.dataset, self.root / "data")
        self.assertEqual(args.max_frames, 16)
        self.assertEqual(args.limit, 0)
        self.assertFalse(args.short_only)
        self.assertFalse(args.use_audio_in_video)
        self.assertEqual(args.task_type, ["Audio Recognition"])
        self.assertEqual(args.cpu_memory, "24GiB")

    def test_bad_config_rejected(self):
        config = self.root / "bad.json"
        for data in ({"max_frame": 24}, {"task_type": "Audio Recognition"},
                     {"limit": True}, {"fps": -1}, {"do_sample": "false"}):
            config.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(config)

    def test_load_and_keep_all_tasks(self):
        self.data["sample"]["task1"] = dict(self.data["sample"]["task0"], answer="A")
        self.write_data()
        samples, videos = load_dataset(self.root)
        self.assertEqual(videos, 1)
        self.assertEqual([s["task_id"] for s in samples], ["task0", "task1"])
        self.assertEqual(samples[0]["video_path"], str(self.root / "videos" / "sample.mp4"))

    def test_short_video_filter_keeps_only_videos_up_to_30_seconds(self):
        self.data["sample"]["video_duration"] = "30s"
        self.write_data()
        samples, _ = load_dataset(self.root)
        self.assertEqual(len(select_samples(samples, 30)), 1)
        self.data["sample"]["video_duration"] = "30.1s"
        self.write_data()
        samples, _ = load_dataset(self.root)
        self.assertEqual(select_samples(samples, 30), [])

    def test_short_video_filter_rejects_nonpositive_limit(self):
        sample = {"duration_seconds": 20}
        with self.assertRaises(ValueError):
            select_samples([sample], 0)

    def test_task_filter_selects_only_requested_tasks(self):
        self.data["sample"]["task1"] = dict(self.data["sample"]["task0"], answer="A")
        self.write_data()
        samples, _ = load_dataset(self.root)
        selected = select_tasks(samples, ["task1"])
        self.assertEqual([sample["task_id"] for sample in selected], ["task1"])
        with self.assertRaises(ValueError):
            select_tasks(samples, ["task-x"])

    def test_domain_and_type_filters_use_intersection(self):
        self.data["sample"]["task0"]["task_domain"] = "Recognition"
        self.data["sample"]["task0"]["task_type"] = "Audio Recognition"
        self.write_data()
        samples, _ = load_dataset(self.root)
        selected = select_metadata(samples, ["Recognition"], ["Audio Recognition"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(select_metadata(samples, ["Reasoning"]), [])

    def test_accuracy_summary_is_grouped_by_category(self):
        rows = [
            {"task_domain": "Recognition", "task_type": "Audio Recognition", "correct": True, "status": "ok"},
            {"task_domain": "Recognition", "task_type": "Audio Recognition", "correct": False, "status": "ok"},
            {"task_domain": "Reasoning", "task_type": "Causal Reasoning", "correct": True, "status": "ok"},
        ]
        self.assertEqual(accuracy_by(rows, "task_domain")["Recognition"],
                         {"correct": 1, "total": 2, "accuracy": 0.5})
        path = self.root / "results" / "grouped.json"
        summary = save_results(path, {}, rows, 3)
        self.assertEqual(summary["accuracy_by_task_domain"]["Reasoning"]["accuracy"], 1.0)
        self.assertIn("Audio Recognition", summary["accuracy_by_task_type"])

    def test_prompt_does_not_leak_answer_or_caption(self):
        sample = load_dataset(self.root)[0][0]
        prompt = build_conversation(sample, 16, 112896)
        changed_answer = build_conversation(dict(sample, answer="A"), 16, 112896)
        self.assertEqual(prompt, changed_answer)
        self.assertNotIn("SECRET HUMAN CAPTION", json.dumps(prompt))
        self.assertIn(sample["question"], prompt[1]["content"][1]["text"])

    def test_missing_video_fails(self):
        (self.root / "videos" / "sample.mp4").unlink()
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.root)

    def test_invalid_gold_fails(self):
        self.data["sample"]["task0"]["answer"] = "D"
        self.write_data()
        with self.assertRaises(ValueError):
            load_dataset(self.root)

    def test_answer_parser_rejects_ambiguity(self):
        options = ["A. Piano", "B. Guitar"]
        for value in ("B", " b. ", "(B)", "Answer: B", "B. Guitar"):
            self.assertEqual(parse_answer(value, options), "B")
        for value in ("", "D", "A or B", "I considered A but chose B"):
            self.assertIsNone(parse_answer(value, options))

    def test_report_counts_errors_in_denominator(self):
        rows = [{"status": "ok", "correct": True},
                {"status": "invalid_answer", "correct": False},
                {"status": "error", "correct": False}]
        path = self.root / "results" / "test.json"
        summary = save_results(path, {}, rows, 3)
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["invalid_answers"], 1)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["results"], rows)


if __name__ == "__main__":
    unittest.main()
