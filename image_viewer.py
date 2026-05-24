#!/usr/bin/env python3
"""Simple image viewer for macOS using OpenCV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".jp2",
    ".pbm",
    ".pgm",
    ".ppm",
    ".sr",
    ".ras",
    ".exr",
    ".hdr",
}

WINDOW_NAME = "Image Viewer"
HELP_TEXT = (
    "Controls: Left/Right or n/p = prev/next | +/- or wheel = zoom | "
    "f = fit | 1 = 100% | o = open | q/Esc = quit"
)


def normalize_extension(path: Path) -> str:
    return path.suffix.lower()


def is_image_file(path: Path) -> bool:
    return path.is_file() and normalize_extension(path) in SUPPORTED_EXTENSIONS


def list_images_in_folder(folder: Path) -> list[Path]:
    images = [p for p in folder.iterdir() if is_image_file(p)]
    images.sort(key=lambda p: p.name.lower())
    return images


def pick_image_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter is not available. Pass an image path on the command line.", file=sys.stderr)
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filetypes = [
        ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif"),
        ("All files", "*.*"),
    ]
    selected = filedialog.askopenfilename(title="Open image", filetypes=filetypes)
    root.destroy()
    if not selected:
        return None
    return Path(selected)


def load_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        background = np.full_like(rgb, 255.0)
        composited = rgb * alpha + background * (1.0 - alpha)
        return composited.astype(np.uint8)

    return image


def fit_scale(image: np.ndarray, max_width: int, max_height: int) -> float:
    height, width = image.shape[:2]
    if width == 0 or height == 0:
        return 1.0
    return min(max_width / width, max_height / height, 1.0)


def render_frame(
    image: np.ndarray,
    scale: float,
    fit_mode: bool,
    window_size: tuple[int, int],
) -> np.ndarray:
    height, width = image.shape[:2]
    if fit_mode:
        scale = fit_scale(image, window_size[0], window_size[1])

    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)

    canvas = np.full((window_size[1], window_size[0], 3), 32, dtype=np.uint8)
    offset_x = max(0, (window_size[0] - target_width) // 2)
    offset_y = max(0, (window_size[1] - target_height) // 2)
    canvas[offset_y : offset_y + target_height, offset_x : offset_x + target_width] = resized
    return canvas


def draw_overlay(
    frame: np.ndarray,
    path: Path,
    image_size: tuple[int, int],
    index: int,
    total: int,
    scale: float,
    fit_mode: bool,
) -> np.ndarray:
    display = frame.copy()
    height, width = image_size
    scale_label = "fit" if fit_mode else f"{scale * 100:.0f}%"
    lines = [
        f"{path.name}  ({index + 1}/{total})",
        f"{width} x {height} px  |  zoom: {scale_label}",
        HELP_TEXT,
    ]

    y = 24
    for line in lines:
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return display


class ImageViewer:
    def __init__(self, initial_path: Path | None) -> None:
        self.window_size = (1280, 800)
        self.scale = 1.0
        self.fit_mode = True
        self.images: list[Path] = []
        self.index = 0
        self.current_image: np.ndarray | None = None

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *self.window_size)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        if initial_path and initial_path.is_file():
            self.open_path(initial_path)
        else:
            chosen = pick_image_file()
            if chosen is None:
                raise SystemExit("No image selected.")
            self.open_path(chosen)

    def open_path(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not is_image_file(path):
            print(f"Unsupported or missing image: {path}", file=sys.stderr)
            return

        self.images = list_images_in_folder(path.parent)
        if not self.images:
            self.images = [path]
        try:
            self.index = self.images.index(path)
        except ValueError:
            self.images.append(path)
            self.images.sort(key=lambda p: p.name.lower())
            self.index = self.images.index(path)

        self._load_current()

    def _load_current(self) -> None:
        path = self.images[self.index]
        image = load_image(path)
        if image is None:
            print(f"Failed to load: {path}", file=sys.stderr)
            self.current_image = None
            return
        self.current_image = image
        self.fit_mode = True

    def _refresh_window_size(self) -> None:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_NAME)
        if width > 0 and height > 0:
            self.window_size = (width, height)

    def _show(self) -> None:
        self._refresh_window_size()
        if self.current_image is None:
            blank = np.full((self.window_size[1], self.window_size[0], 3), 32, dtype=np.uint8)
            cv2.imshow(WINDOW_NAME, blank)
            return

        height, width = self.current_image.shape[:2]
        frame = render_frame(self.current_image, self.scale, self.fit_mode, self.window_size)
        overlay = draw_overlay(
            frame,
            self.images[self.index],
            (height, width),
            self.index,
            len(self.images),
            self.scale,
            self.fit_mode,
        )
        cv2.imshow(WINDOW_NAME, overlay)

    def _on_mouse(self, event: int, _x: int, _y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self._adjust_zoom(0.1 * delta)

    def _adjust_zoom(self, delta: float) -> None:
        self.fit_mode = False
        self.scale = max(0.05, min(8.0, self.scale + delta))

    def _move(self, step: int) -> None:
        if len(self.images) <= 1:
            return
        self.index = (self.index + step) % len(self.images)
        self._load_current()

    def run(self) -> None:
        while True:
            self._show()
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue

            normalized = key & 0xFF
            if normalized in (ord("q"), 27):
                break
            if normalized in (ord("n"), 83, 63235):
                self._move(1)
            elif normalized in (ord("p"), 81, 63234):
                self._move(-1)
            elif normalized in (ord("+"), ord("=")):
                self._adjust_zoom(0.1)
            elif normalized in (ord("-"), ord("_")):
                self._adjust_zoom(-0.1)
            elif normalized == ord("f"):
                self.fit_mode = True
            elif normalized == ord("1"):
                self.fit_mode = False
                self.scale = 1.0
            elif normalized == ord("o"):
                chosen = pick_image_file()
                if chosen is not None:
                    self.open_path(chosen)

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View JPEG, PNG, and other common image formats.")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an image file. Opens a file picker when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial = Path(args.image).expanduser() if args.image else None
    viewer = ImageViewer(initial)
    viewer.run()


if __name__ == "__main__":
    main()
