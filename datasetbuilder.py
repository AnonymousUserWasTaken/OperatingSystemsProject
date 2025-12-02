import time
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

import cv2
import numpy as np
import mss
from pynput import mouse


@dataclass
class ClickEvent:
    t: float
    x: int
    y: int
    button: str


@dataclass
class FrameRecord:
    t: float
    img: np.ndarray


class ClickRecorder:
    """
    Records screen frames AND global mouse clicks, then produces:
    - Frame sequence
    - CLICK! synthetic frames
    - Final MP4 video
    - clicks.json metadata

    The metadata now includes, for each CLICK! frame:
        - the frame index in the final video sequence
        - the source screen frame index
        - click coordinates in screen space and frame space (raw + normalized)
    """

    def __init__(
        self,
        out_dir: str,
        fps: int,
        duration_sec: int,
        monitor_index: int,
        mp4_name: str,
        session_dir: bool = True,
        session_name: str = None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if session_dir:
            self.session_name = session_name if session_name else f"session_{timestamp}"
            self.full_out_dir = os.path.join(out_dir, self.session_name)
        else:
            self.full_out_dir = out_dir

        os.makedirs(self.full_out_dir, exist_ok=True)

        self.fps = fps
        self.duration_sec = duration_sec
        self.monitor_index = monitor_index
        self.mp4_name = mp4_name

        self.frames: List[FrameRecord] = []
        self.clicks: List[ClickEvent] = []

    # ------------- Mouse Hook ------------------
    def _on_click(self, x, y, button, pressed):
        if pressed:
            self.clicks.append(
                ClickEvent(
                    t=time.time(),
                    x=int(x),
                    y=int(y),
                    button=str(button),
                )
            )

    # ------------- Main Record Loop ------------------
    def record(self):
        print(f"[Recorder] Starting capture: {self.duration_sec}s @ {self.fps} FPS")
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]

            listener = mouse.Listener(on_click=self._on_click)
            listener.start()

            start_t = time.time()
            next_frame_t = start_t
            interval = 1.0 / self.fps

            try:
                while True:
                    now = time.time()
                    if now - start_t >= self.duration_sec:
                        break

                    if now >= next_frame_t:
                        sct_img = sct.grab(monitor)
                        img = np.array(sct_img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                        self.frames.append(FrameRecord(t=now, img=img))
                        next_frame_t += interval

            except KeyboardInterrupt:
                print("[Recorder] Interrupted manually.")

            finally:
                listener.stop()

        print(f"[Recorder] Recorded {len(self.frames)} frames and {len(self.clicks)} clicks.")
        self._postprocess_and_save(monitor)

    # ------------- Match clicks to frame index ------------------
    def _assign_clicks_to_frames(self) -> Dict[int, List[ClickEvent]]:
        """
        For each click event, assign it to the closest recorded frame index
        based on timestamp.
        """
        frame_times = np.array([f.t for f in self.frames])
        mapping: Dict[int, List[ClickEvent]] = {}

        for c in self.clicks:
            idx = int(np.argmin(np.abs(frame_times - c.t)))
            mapping.setdefault(idx, []).append(c)

        return mapping

    # ------------- CLICK! Label Frame ------------------
    def _make_click_label_frame(self, width, height, click: ClickEvent, monitor: Dict[str, Any]):
        """
        Create a synthetic CLICK! frame on a gray background,
        marking where the click occurred.

        Returns:
            img: np.ndarray (H, W, 3)
            info: dict with click position metadata, e.g.
                {
                    "screen_x": ...,
                    "screen_y": ...,
                    "screen_x_norm": ...,
                    "screen_y_norm": ...,
                    "frame_x": ...,
                    "frame_y": ...,
                    "frame_x_norm": ...,
                    "frame_y_norm": ...
                }
        """
        # Base gray background
        img = np.full((height, width, 3), 128, dtype=np.uint8)

        # Centered CLICK! text
        text = "CLICK!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3.0
        thick = 6
        size, _ = cv2.getTextSize(text, font, scale, thick)
        tw, th = size
        tx = (width - tw) // 2
        ty = (height + th) // 2
        cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # Map screen coordinates -> frame coordinates
        mon_w, mon_h = monitor["width"], monitor["height"]

        # Screen-space normalized coords (0..1)
        screen_x_norm = click.x / float(mon_w)
        screen_y_norm = click.y / float(mon_h)

        # Frame-space pixel coords (using normalized screen coords)
        cx = int(screen_x_norm * width)
        cy = int(screen_y_norm * height)

        # Frame-space normalized coords (0..1)
        frame_x_norm = cx / float(width)
        frame_y_norm = cy / float(height)

        # Draw crosshair at click location
        cv2.circle(img, (cx, cy), 25, (0, 0, 255), 4)
        cv2.line(img, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 4)
        cv2.line(img, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 4)

        info = {
            "screen_x": int(click.x),
            "screen_y": int(click.y),
            "screen_x_norm": float(screen_x_norm),
            "screen_y_norm": float(screen_y_norm),
            "frame_x": int(cx),
            "frame_y": int(cy),
            "frame_x_norm": float(frame_x_norm),
            "frame_y_norm": float(frame_y_norm),
        }

        return img, info

    # ------------- Save Frames + MP4 + Metadata ------------------
    def _postprocess_and_save(self, monitor: Dict[str, Any]):
        click_map = self._assign_clicks_to_frames()

        final_frames: List[np.ndarray] = []
        frame_meta: List[Dict[str, Any]] = []

        if not self.frames:
            print("[Recorder] No frames captured, nothing to save.")
            return

        h, w, _ = self.frames[0].img.shape

        # Build final frame sequence and metadata
        for src_idx, fr in enumerate(self.frames):
            # Add the original frame
            final_frames.append(fr.img)
            frame_idx = len(final_frames) - 1
            frame_meta.append(
                {
                    "type": "frame",
                    "frame_idx": frame_idx,      # index in final_frames / MP4 / PNG filenames
                    "src_idx": src_idx,          # index in original capture sequence
                    "t": fr.t,
                }
            )

            # If there were clicks mapped to this frame, add CLICK! label frames
            if src_idx in click_map:
                for c in click_map[src_idx]:
                    lbl_img, click_info = self._make_click_label_frame(w, h, c, monitor)
                    final_frames.append(lbl_img)
                    click_frame_idx = len(final_frames) - 1

                    # one meta entry per CLICK! frame, with full click info
                    meta_entry = {
                        "type": "click_label",
                        "frame_idx": click_frame_idx,  # in final_frames / video
                        "src_idx": src_idx,            # which original frame this click belongs to
                        "t": c.t,
                        "button": c.button,
                        # click positions
                        "click_screen_x": click_info["screen_x"],
                        "click_screen_y": click_info["screen_y"],
                        "click_screen_x_norm": click_info["screen_x_norm"],
                        "click_screen_y_norm": click_info["screen_y_norm"],
                        "click_frame_x": click_info["frame_x"],
                        "click_frame_y": click_info["frame_y"],
                        "click_frame_x_norm": click_info["frame_x_norm"],
                        "click_frame_y_norm": click_info["frame_y_norm"],
                    }
                    frame_meta.append(meta_entry)

        # -------- Save PNG frames (optional, but handy for training) --------
        for i, img in enumerate(final_frames):
            out_path = os.path.join(self.full_out_dir, f"frame_{i:06d}.png")
            cv2.imwrite(out_path, img)

        # -------- Save MP4 --------
        mp4_path = os.path.join(self.full_out_dir, self.mp4_name)
        print(f"[Recorder] Writing video to {mp4_path}")

        out = cv2.VideoWriter(
            mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h),
        )
        for img in final_frames:
            out.write(img)
        out.release()

        # -------- Metadata --------
        clicks_plain = [c.__dict__ for c in self.clicks]

        meta = {
            "fps": self.fps,
            "duration": self.duration_sec,
            "num_raw_frames": len(self.frames),
            "num_final_frames": len(final_frames),
            "monitor": {
                "width": monitor["width"],
                "height": monitor["height"],
                "left": monitor.get("left", 0),
                "top": monitor.get("top", 0),
            },
            "clicks": clicks_plain,     # raw click events
            "frame_meta": frame_meta,   # per-frame supervisor data
            "video_name": self.mp4_name,
            "session_dir": self.full_out_dir,
        }

        meta_path = os.path.join(self.full_out_dir, "clicks.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Recorder] Dataset complete. Saved metadata to {meta_path}")


# =====================================================
#                     ARGPARSE
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UI Click Recorder")

    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="output_default")
    parser.add_argument("--mp4_name", type=str, default="output_default.mp4")

    parser.add_argument(
        "--session_dir",
        action="store_true",
        default=True,
        help="Place output inside a timestamped session folder.",
    )
    parser.add_argument(
        "--session_name",
        type=str,
        default=None,
        help="Optional custom name for the session folder.",
    )

    args = parser.parse_args()

    recorder = ClickRecorder(
        out_dir=args.out_dir,
        fps=args.fps,
        duration_sec=args.duration,
        monitor_index=args.monitor,
        mp4_name=args.mp4_name,
        session_dir=args.session_dir,
        session_name=args.session_name,
    )

    recorder.record()
