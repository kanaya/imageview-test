#!/usr/bin/env python3
"""Simple image viewer for macOS using OpenCV."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
CIRCLE_RADIUS = 40
MORPH_KERNEL_SIZE = 3
HELP_TEXT = (
    "Controls: click = Laplacian in circle | Left/Right or n/p = prev/next | "
    "+/- or wheel = zoom | f = fit | 1 = 100% | o = open | q/Esc = quit"
)


@dataclass(frozen=True)
class DisplayLayout:
    scale: float
    offset_x: int
    offset_y: int
    display_width: int
    display_height: int
    image_width: int
    image_height: int


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


def compute_display_layout(
    image: np.ndarray,
    scale: float,
    fit_mode: bool,
    window_size: tuple[int, int],
) -> DisplayLayout:
    height, width = image.shape[:2]
    if fit_mode:
        scale = fit_scale(image, window_size[0], window_size[1])

    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))
    offset_x = max(0, (window_size[0] - display_width) // 2)
    offset_y = max(0, (window_size[1] - display_height) // 2)
    return DisplayLayout(
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        display_width=display_width,
        display_height=display_height,
        image_width=width,
        image_height=height,
    )


def window_point_to_image(x: int, y: int, layout: DisplayLayout) -> tuple[int, int] | None:
    local_x = x - layout.offset_x
    local_y = y - layout.offset_y
    if not (0 <= local_x < layout.display_width and 0 <= local_y < layout.display_height):
        return None

    image_x = int(round(local_x / layout.scale))
    image_y = int(round(local_y / layout.scale))
    image_x = max(0, min(layout.image_width - 1, image_x))
    image_y = max(0, min(layout.image_height - 1, image_y))
    return image_x, image_y


def denoise_laplacian(laplacian_gray: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    eroded = cv2.erode(laplacian_gray, kernel)
    return cv2.dilate(eroded, kernel)


def normalize_max_to_255(gray: np.ndarray) -> np.ndarray:
    max_val = int(gray.max())
    if max_val <= 0 or max_val == 255:
        return gray
    scaled = gray.astype(np.float32) * (255.0 / max_val)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_laplacian_in_circles(
    image: np.ndarray,
    centers: list[tuple[int, int]],
    radius: int = CIRCLE_RADIUS,
) -> np.ndarray:
    if not centers:
        return image.copy()

    display = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian)
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    laplacian_denoised = denoise_laplacian(laplacian_norm)
    laplacian_denoised = normalize_max_to_255(laplacian_denoised)
    laplacian_bgr = cv2.cvtColor(laplacian_denoised, cv2.COLOR_GRAY2BGR)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for center_x, center_y in centers:
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    display[mask == 255] = laplacian_bgr[mask == 255]
    return display


def render_frame(
    image: np.ndarray,
    scale: float,
    fit_mode: bool,
    window_size: tuple[int, int],
) -> tuple[np.ndarray, DisplayLayout]:
    layout = compute_display_layout(image, scale, fit_mode, window_size)

    interpolation = cv2.INTER_AREA if layout.scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (layout.display_width, layout.display_height),
        interpolation=interpolation,
    )

    canvas = np.full((window_size[1], window_size[0], 3), 32, dtype=np.uint8)
    canvas[
        layout.offset_y : layout.offset_y + layout.display_height,
        layout.offset_x : layout.offset_x + layout.display_width,
    ] = resized

    return canvas, layout


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
        self.base_image: np.ndarray | None = None
        self.circle_marks: dict[str, list[tuple[int, int]]] = {}
        self.display_layout: DisplayLayout | None = None

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
            self.base_image = None
            return
        self.base_image = image
        self.fit_mode = True

    def _refresh_window_size(self) -> None:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_NAME)
        if width > 0 and height > 0:
            self.window_size = (width, height)

    def _show(self) -> None:
        self._refresh_window_size()
        if self.base_image is None:
            blank = np.full((self.window_size[1], self.window_size[0], 3), 32, dtype=np.uint8)
            cv2.imshow(WINDOW_NAME, blank)
            return

        height, width = self.base_image.shape[:2]
        image_key = str(self.images[self.index])
        circle_centers = self.circle_marks.get(image_key, [])
        display_image = apply_laplacian_in_circles(self.base_image, circle_centers)
        frame, self.display_layout = render_frame(
            display_image,
            self.scale,
            self.fit_mode,
            self.window_size,
        )
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

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._add_circle_at(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self._adjust_zoom(0.1 * delta)

    def _add_circle_at(self, x: int, y: int) -> None:
        if self.base_image is None or self.display_layout is None:
            return

        image_point = window_point_to_image(x, y, self.display_layout)
        if image_point is None:
            return

        image_key = str(self.images[self.index])
        marks = self.circle_marks.setdefault(image_key, [])
        marks.append(image_point)

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
