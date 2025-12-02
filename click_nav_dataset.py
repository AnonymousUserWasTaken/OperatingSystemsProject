# ============================================================
# FILE: ShowUI/model/showui/click_nav_dataset.py
# ============================================================

import os
from typing import List, Dict, Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from .processing_showui import ShowUIProcessor


DATA_ROOT = os.path.join(os.path.dirname(__file__), "OSProject")

# Hard cap to keep VRAM sane
MAX_FRAMES = 32


class ClickNavVideoDataset(Dataset):
    """
    Dataset loader for Click Navigation training data.

    Expected directory structure:
    OSProject/
        train/
            MySessionName/
                frame_000000.png
                frame_000001.png
                ...
                clicks.json (optional, for future supervised click labels)
        val/
        test/
    """

    def __init__(self, split: str, processor: "ShowUIProcessor"):
        self.split_dir = os.path.join(DATA_ROOT, split)
        self.processor = processor
        self.examples = self._find_all_sessions()

    def _find_all_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        if not os.path.isdir(self.split_dir):
            return sessions

        for sess in os.listdir(self.split_dir):
            sess_dir = os.path.join(self.split_dir, sess)
            if not os.path.isdir(sess_dir):
                continue

            frame_files = sorted(
                f for f in os.listdir(sess_dir)
                if f.startswith("frame_") and f.endswith(".png")
            )
            if not frame_files:
                continue

            sessions.append({
                "name": sess,
                "dir": sess_dir,
                "frames": frame_files,
            })

        return sessions

    def __len__(self):
        return len(self.examples)

    def _load_frames(self, sess) -> List[Image.Image]:
        out = []
        for fname in sess["frames"]:
            path = os.path.join(sess["dir"], fname)
            out.append(Image.open(path).convert("RGB"))
        return out

    def _subsample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """
        Uniformly subsample frames to at most MAX_FRAMES to avoid OOM.

        If there are <= MAX_FRAMES, return as-is.
        If more, pick roughly evenly spaced frames.
        """
        n = len(frames)
        if n <= MAX_FRAMES:
            return frames

        # stride to get ~MAX_FRAMES frames
        stride = max(1, n // MAX_FRAMES)
        sampled = [frames[i] for i in range(0, n, stride)]
        # Just in case rounding gave us > MAX_FRAMES
        return sampled[:MAX_FRAMES]

    def _make_instruction(self, sess_name: str) -> str:
        """
        Example: 'open_steam_001' → 'open steam'
        """
        tokens = sess_name.split("_")
        tokens = [t for t in tokens if not t.isdigit()]
        return " ".join(tokens)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sess = self.examples[idx]

        frames = self._load_frames(sess)
        frames = self._subsample_frames(frames)

        instruction = self._make_instruction(sess["name"])

        proc = self.processor(
            text=instruction,
            videos=[frames],
            return_tensors="pt",
            training=True,   # important so ui-graph behavior matches train mode
        )

        batch = {
            "input_ids": proc["input_ids"].squeeze(0),
            "attention_mask": proc["attention_mask"].squeeze(0),
            "pixel_values_videos": proc["pixel_values_videos"].squeeze(0),
            "video_grid_thw": proc["video_grid_thw"].squeeze(0),
            "labels": proc["input_ids"].squeeze(0).clone(),
        }

        return batch


# ------------------------------------------------------------
# Collate function
# ------------------------------------------------------------
def click_nav_collate(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stacks already-padded tensors.
    """

    out: Dict[str, torch.Tensor] = {}

    for key in ["input_ids", "attention_mask", "labels"]:
        out[key] = torch.stack([ex[key] for ex in examples], dim=0)

    out["pixel_values_videos"] = torch.stack(
        [ex["pixel_values_videos"] for ex in examples], dim=0
    )

    out["video_grid_thw"] = torch.stack(
        [ex["video_grid_thw"] for ex in examples], dim=0
    )

    return out



# ------------------------------------------------------------
# Collate function
# ------------------------------------------------------------
def click_nav_collate(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate already-processed examples into a batch.

    We assume ShowUIProcessor already did any padding/truncation needed
    across tokens and video patches. So we can just stack.
    """
    out: Dict[str, torch.Tensor] = {}

    for key in ["input_ids", "attention_mask", "labels"]:
        out[key] = torch.stack([ex[key] for ex in examples], dim=0)

    out["pixel_values_videos"] = torch.stack(
        [ex["pixel_values_videos"] for ex in examples], dim=0
    )

    out["video_grid_thw"] = torch.stack(
        [ex["video_grid_thw"] for ex in examples], dim=0
    )

    return out
