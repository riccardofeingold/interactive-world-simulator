#!/usr/bin/env python3
"""Convert OrcaHand video annotations into IWS HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
from tqdm import tqdm

DEFAULT_SOURCE_DIR = Path(
    "/home/riccardo/OrcaHandWorldModel/datasets/2026-03-14T13-34-49/"
    "large_real_dataset_5fps_135_240"
)
DEFAULT_META_DIR = Path(
    "/data/OrcaHandWorldModel/dataset_meta_info/2026-03-14T13-34-49/"
    "large_real_dataset_5fps_135_240"
)
FALLBACK_META_DIR = Path(
    "/home/riccardo/OrcaHandWorldModel/dataset_meta_info/2026-03-14T13-34-49/"
    "large_real_dataset_5fps_135_240"
)
DEFAULT_OUTPUT_DIR = Path("data/orcahand/large_real_dataset_5fps_135_240")
FPS = 5.0
CARTESIAN_DIM = 6
HAND_DIM = 17
ACTION_DIM = CARTESIAN_DIM + HAND_DIM
SAMPLE_INDEX_NAME = "sample_index.json"


@dataclass(frozen=True)
class EpisodeSpec:
    annotation_path: Path
    source_episode_id: int
    split: str
    output_episode_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OrcaHand annotations/videos to IWS HDF5 episodes."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--train-sample-file", type=Path, default=None)
    parser.add_argument("--val-sample-file", type=Path, default=None)
    parser.add_argument(
        "--ignore-sample-files",
        action="store_true",
        help="Ignore train/val sample JSON files and use deterministic episode splitting.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=["camera_0_color", "camera_1_color"],
        help="Camera dataset names, mapped in order to annotation['videos'] entries.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.",
    )
    return parser.parse_args()


def load_annotation(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_json(path: Path) -> object:
    with path.open("r") as f:
        return json.load(f)


def sorted_annotation_paths(source_dir: Path) -> list[Path]:
    ann_dir = source_dir / "annotation"
    paths = sorted(ann_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(f"No annotation JSON files found in {ann_dir}")
    return paths


def split_annotations(
    paths: Sequence[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1), got {val_ratio}")
    paths = list(paths)
    rng = np.random.default_rng(seed)
    order = np.arange(len(paths))
    rng.shuffle(order)
    if len(paths) <= 1 or val_ratio == 0.0:
        val_count = 0
    else:
        val_count = max(1, int(round(len(paths) * val_ratio)))
        val_count = min(val_count, len(paths) - 1)
    val_idx = set(order[:val_count].tolist())
    train_paths = [p for i, p in enumerate(paths) if i not in val_idx]
    val_paths = [p for i, p in enumerate(paths) if i in val_idx]
    return train_paths, val_paths


def validate_sample_list(samples: object, split_name: str) -> list[dict]:
    if not isinstance(samples, list):
        raise ValueError(f"{split_name} sample file must contain a list")
    normalized = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{split_name}[{idx}] must be an object")
        if "episode_id" not in sample or "frame_ids" not in sample:
            raise ValueError(f"{split_name}[{idx}] must contain episode_id and frame_ids")
        frame_ids = sample["frame_ids"]
        if not isinstance(frame_ids, list) or len(frame_ids) != 1:
            raise ValueError(f"{split_name}[{idx}].frame_ids must be a one-item list")
        normalized.append(
            {
                "episode_id": int(sample["episode_id"]),
                "frame_id": int(frame_ids[0]),
            }
        )
    return normalized


def unique_episode_ids(samples: Sequence[dict]) -> list[int]:
    seen = set()
    ids = []
    for sample in samples:
        episode_id = sample["episode_id"]
        if episode_id not in seen:
            seen.add(episode_id)
            ids.append(episode_id)
    return ids


def resolve_sample_files(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if args.ignore_sample_files:
        return None, None
    train_file = args.train_sample_file or args.meta_dir / "train_sample.json"
    val_file = args.val_sample_file or args.meta_dir / "val_sample.json"
    if train_file.exists() and val_file.exists():
        return train_file, val_file
    if args.train_sample_file is None and args.val_sample_file is None:
        fallback_train = FALLBACK_META_DIR / "train_sample.json"
        fallback_val = FALLBACK_META_DIR / "val_sample.json"
        if fallback_train.exists() and fallback_val.exists():
            return fallback_train, fallback_val
    if args.train_sample_file is not None or args.val_sample_file is not None:
        missing = [str(p) for p in (train_file, val_file) if not p.exists()]
        raise FileNotFoundError(f"Missing requested sample split file(s): {missing}")
    return None, None


def build_specs_from_sample_files(
    source_dir: Path,
    train_file: Path,
    val_file: Path,
    limit: int | None,
) -> tuple[list[EpisodeSpec], dict[str, list[dict]]]:
    split_samples = {
        "train": validate_sample_list(load_json(train_file), "train"),
        "val": validate_sample_list(load_json(val_file), "val"),
    }
    if limit is not None:
        split_samples = {
            split: samples[:limit]
            for split, samples in split_samples.items()
        }

    specs = []
    sample_indices = {}
    for split, samples in split_samples.items():
        episode_ids = unique_episode_ids(samples)
        source_to_output = {source_id: out_id for out_id, source_id in enumerate(episode_ids)}
        for source_id, out_id in source_to_output.items():
            annotation_path = source_dir / "annotation" / f"{source_id}.json"
            if not annotation_path.exists():
                raise FileNotFoundError(annotation_path)
            specs.append(
                EpisodeSpec(
                    annotation_path=annotation_path,
                    source_episode_id=source_id,
                    split=split,
                    output_episode_id=out_id,
                )
            )
        sample_indices[split] = [
            {
                "episode_id": source_to_output[sample["episode_id"]],
                "source_episode_id": sample["episode_id"],
                "frame_id": sample["frame_id"],
            }
            for sample in samples
        ]
    return specs, sample_indices


def as_2d_float_array(value: list, key: str, expected_dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"{key} must have shape (T, {expected_dim}), got {arr.shape}")
    return arr


def video_metadata(video_path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return frame_count, width, height, fps


def validate_episode(annotation: dict, source_dir: Path, camera_keys: Sequence[str]) -> dict:
    cartesian = as_2d_float_array(
        annotation["observation.state.cartesian_position"], "observation.state.cartesian_position", CARTESIAN_DIM
    )
    hand = as_2d_float_array(
        annotation["action.hand_joint_position"], "action.hand_joint_position", HAND_DIM
    )
    if cartesian.shape[0] != hand.shape[0]:
        raise ValueError(
            "Action lengths differ: "
            f"cartesian={cartesian.shape[0]}, hand={hand.shape[0]}"
        )

    # SWAPING thumb_abd with thumb_mcp => SIM_DATA Convetion and how WM1_midtraining and WM2_svd has been trained
    hand[:, [1, 2]] = hand[:, [2, 1]]
    action = np.concatenate([cartesian, hand], axis=1).astype(np.float32)
    if action.shape[1] != ACTION_DIM:
        raise ValueError(f"action must have dim {ACTION_DIM}, got {action.shape[1]}")

    obs_cartesian = as_2d_float_array(
        annotation["observation.state.cartesian_position"],
        "observation.state.cartesian_position",
        CARTESIAN_DIM,
    )
    obs_hand = as_2d_float_array(
        annotation["observation.state.hand_joint_position"],
        "observation.state.hand_joint_position",
        HAND_DIM,
    )
    if obs_cartesian.shape[0] != action.shape[0] or obs_hand.shape[0] != action.shape[0]:
        raise ValueError("Observation and action lengths do not match")

    videos = annotation.get("videos", [])
    if len(videos) != len(camera_keys):
        raise ValueError(
            f"Expected {len(camera_keys)} videos for {camera_keys}, got {len(videos)}"
        )

    video_infos = []
    for video_entry in videos:
        rel_path = Path(video_entry["video_path"])
        if not rel_path.parts or rel_path.parts[0] != "videos":
            raise ValueError(f"Expected regular videos/ path, got {rel_path}")
        video_path = source_dir / rel_path
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        frame_count, width, height, fps = video_metadata(video_path)
        if frame_count != action.shape[0]:
            raise ValueError(
                f"Frame/action mismatch for {video_path}: {frame_count} vs {action.shape[0]}"
            )
        video_infos.append(
            {
                "path": video_path,
                "relative_path": str(rel_path),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
            }
        )

    return {
        "action": action,
        "obs_cartesian": obs_cartesian.astype(np.float32),
        "obs_hand": obs_hand.astype(np.float32),
        "video_infos": video_infos,
        "length": action.shape[0],
    }


def read_video_rgb(video_path: Path, expected_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        for _ in range(expected_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Failed reading frame {len(frames)} from {video_path}")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ok, _ = cap.read()
        if ok:
            raise RuntimeError(f"Video {video_path} has more than {expected_frames} frames")
    finally:
        cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)


def write_episode(
    output_path: Path,
    annotation: dict,
    validated: dict,
    camera_keys: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["source_episode_id"] = int(annotation["episode_id"])
        f.attrs["source_success"] = int(annotation.get("success", 1))
        f.attrs["source_video_length"] = int(annotation.get("video_length", validated["length"]))
        f.create_dataset("action", data=validated["action"], dtype="float32")
        f.create_dataset(
            "timestamp",
            data=np.arange(validated["length"], dtype=np.float64) / FPS,
            dtype="float64",
        )
        obs = f.create_group("obs")
        obs.create_dataset("cartesian_pose", data=validated["obs_cartesian"], dtype="float32")
        obs.create_dataset(
            "hand_joint_positions", data=validated["obs_hand"], dtype="float32"
        )
        images = obs.create_group("images")
        for camera_key, info in zip(camera_keys, validated["video_infos"], strict=True):
            frames = read_video_rgb(info["path"], validated["length"])
            images.create_dataset(
                camera_key,
                data=frames,
                dtype="uint8",
                compression="gzip",
                compression_opts=4,
                chunks=(1, frames.shape[1], frames.shape[2], frames.shape[3]),
            )


def make_specs(train_paths: Sequence[Path], val_paths: Sequence[Path]) -> list[EpisodeSpec]:
    specs = []
    for split, paths in (("train", train_paths), ("val", val_paths)):
        for out_id, path in enumerate(paths):
            specs.append(
                EpisodeSpec(
                    annotation_path=path,
                    source_episode_id=int(path.stem),
                    split=split,
                    output_episode_id=out_id,
                )
            )
    return specs


def print_dry_run(
    specs: Sequence[EpisodeSpec],
    source_dir: Path,
    camera_keys: Sequence[str],
    sample_indices: dict[str, list[dict]] | None,
) -> None:
    print(f"source_dir: {source_dir}")
    print("episode counts:", {split: sum(s.split == split for s in specs) for split in ("train", "val")})
    if sample_indices is not None:
        print("sample counts:", {split: len(samples) for split, samples in sample_indices.items()})
    for spec in specs:
        ann = load_annotation(spec.annotation_path)
        validated = validate_episode(ann, source_dir, camera_keys)
        videos = ", ".join(
            f"{info['relative_path']}:{info['frame_count']}f@{info['width']}x{info['height']}"
            for info in validated["video_infos"]
        )
        print(
            f"{spec.split}/episode_{spec.output_episode_id}.hdf5 "
            f"<- source {spec.source_episode_id}: T={validated['length']} "
            f"action_dim={validated['action'].shape[1]} videos=[{videos}]"
        )


def write_sample_indices(output_dir: Path, sample_indices: dict[str, list[dict]] | None) -> None:
    if sample_indices is None:
        return
    for split, samples in sample_indices.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / SAMPLE_INDEX_NAME).open("w") as f:
            json.dump(samples, f, indent=2)


def validate_sample_indices(
    specs: Sequence[EpisodeSpec],
    sample_indices: dict[str, list[dict]] | None,
    episode_lengths: dict[tuple[str, int], int],
) -> None:
    if sample_indices is None:
        return
    spec_keys = {(spec.split, spec.output_episode_id) for spec in specs}
    for split, samples in sample_indices.items():
        for idx, sample in enumerate(samples):
            key = (split, int(sample["episode_id"]))
            if key not in spec_keys:
                raise ValueError(f"{split} sample {idx} references missing episode {key}")
            frame_id = int(sample["frame_id"])
            if frame_id < 0 or frame_id >= episode_lengths[key]:
                raise ValueError(
                    f"{split} sample {idx} frame_id={frame_id} outside episode length "
                    f"{episode_lengths[key]}"
                )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if len(args.camera_keys) != 2:
        raise ValueError("This converter expects exactly two camera keys.")

    train_sample_file, val_sample_file = resolve_sample_files(args)
    sample_indices = None
    if train_sample_file is not None and val_sample_file is not None:
        specs, sample_indices = build_specs_from_sample_files(
            source_dir, train_sample_file, val_sample_file, args.limit
        )
        print(f"Using sample split files: {train_sample_file} and {val_sample_file}")
    else:
        paths = sorted_annotation_paths(source_dir)
        if args.limit is not None:
            paths = paths[: args.limit]
        train_paths, val_paths = split_annotations(paths, args.val_ratio, args.seed)
        specs = make_specs(train_paths, val_paths)
        print("Using deterministic episode split; no sample split files found.")

    if args.dry_run:
        print_dry_run(specs, source_dir, args.camera_keys, sample_indices)
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    episode_lengths = {}
    for spec in tqdm(specs, desc="Converting episodes"):
        ann = load_annotation(spec.annotation_path)
        validated = validate_episode(ann, source_dir, args.camera_keys)
        episode_lengths[(spec.split, spec.output_episode_id)] = validated["length"]
        output_path = output_dir / spec.split / f"episode_{spec.output_episode_id}.hdf5"
        write_episode(output_path, ann, validated, args.camera_keys)

    validate_sample_indices(specs, sample_indices, episode_lengths)
    write_sample_indices(output_dir, sample_indices)

    print(f"Wrote {sum(s.split == 'train' for s in specs)} train episodes")
    print(f"Wrote {sum(s.split == 'val' for s in specs)} val episodes")
    if sample_indices is not None:
        print(f"Wrote {len(sample_indices.get('train', []))} train sample indices")
        print(f"Wrote {len(sample_indices.get('val', []))} val sample indices")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
