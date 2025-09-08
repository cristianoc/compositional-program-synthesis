#!/usr/bin/env python3
"""
Image Generation Utilities
Shared utilities for rendering ARC-style grid puzzles as images.
"""

import numpy as np
from typing import Dict, Tuple


# Standard ARC color palette
PALETTE = {
    0: (0, 0, 0),      # Black
    1: (0, 0, 255),    # Blue
    2: (255, 0, 0),    # Red
    3: (0, 255, 0),    # Green
    4: (255, 255, 0),  # Yellow
    5: (128, 128, 128), # Gray
    6: (255, 192, 203), # Pink
    7: (255, 165, 0),   # Orange
    8: (0, 128, 128),   # Teal
    9: (139, 69, 19)    # Brown
}

# Common colors
YELLOW = (255, 255, 0)


def grid_to_rgb(g: np.ndarray, palette: Dict[int, Tuple[int, int, int]] = PALETTE) -> np.ndarray:
    """Convert a grid of color indices to RGB image."""
    h, w = g.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb[i, j, :] = palette.get(int(g[i, j]), (0, 0, 0))
    return rgb


def upsample(img: np.ndarray, scale: int) -> np.ndarray:
    """Upsample an image by repeating pixels."""
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def save_png(path: str, rgb: np.ndarray):
    """Save RGB image as uncompressed PNG."""
    import struct
    import binascii
    
    H, W, C = rgb.shape
    assert C == 3
    
    # Build raw bytes with no filter per row
    raw = b''.join(b"\x00" + rgb[i].tobytes() for i in range(H))

    # Deterministic zlib stream with stored (uncompressed) deflate blocks
    def adler32(data: bytes) -> int:
        MOD = 65521
        a = 1
        b = 0
        for byte in data:
            a = (a + byte) % MOD
            b = (b + a) % MOD
        return (b << 16) | a

    def zlib_stored(data: bytes) -> bytes:
        # zlib header: 0x78 0x01 (CMF=0x78, FLG=0x01) satisfies 31-check and FLEVEL=0
        header = b"\x78\x01"
        out = [header]
        i = 0
        n = len(data)
        while i < n:
            chunk = data[i : i + 65535]
            i += len(chunk)
            bfinal = 1 if i >= n else 0
            # Stored block header: 3 bits (BFINAL, BTYPE=00) → at byte boundary => byte is 0x01 for final else 0x00
            out.append(bytes([bfinal]))
            L = len(chunk)
            out.append(L.to_bytes(2, 'little'))
            out.append((0xFFFF - L).to_bytes(2, 'little'))
            out.append(chunk)
        out.append(adler32(data).to_bytes(4, 'big'))
        return b''.join(out)

    # PNG chunks
    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", binascii.crc32(typ + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    idat = zlib_stored(raw)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def draw_rect_outline(img: np.ndarray, y1: int, x1: int, y2: int, x2: int, 
                     color: Tuple[int, int, int] = YELLOW, scale: int = 16):
    """Draw a rectangle outline on an upscaled image."""
    h, w, _ = img.shape
    # Draw horizontal lines
    for x in range(x1*scale, min((x2+1)*scale, w)):
        if 0 <= y1*scale < h: 
            img[y1*scale, x, :] = color
        if 0 <= y2*scale < h: 
            img[y2*scale, x, :] = color
    # Draw vertical lines
    for y in range(y1*scale, min((y2+1)*scale, h)):
        if 0 <= x1*scale < w: 
            img[y, x1*scale, :] = color
        if 0 <= x2*scale < w: 
            img[y, x2*scale, :] = color


def _font_3x5():
    """3x5 pixel font for text rendering."""
    F = {
        ' ': ["000","000","000","000","000"],
        'A': ["010","101","111","101","101"],
        'B': ["110","101","110","101","110"],
        'C': ["011","100","100","100","011"],
        'D': ["110","101","101","101","110"],
        'E': ["111","100","110","100","111"],
        'F': ["111","100","110","100","100"],
        'H': ["101","101","111","101","101"],
        'I': ["111","010","010","010","111"],
        'J': ["111","001","001","101","010"],
        'L': ["100","100","100","100","111"],
        'N': ["101","111","111","111","101"],
        'O': ["010","101","101","101","010"],
        'P': ["110","101","110","100","100"],
        'R': ["110","101","110","101","101"],
        'S': ["011","100","010","001","110"],
        'T': ["111","010","010","010","010"],
        'U': ["101","101","101","101","111"],
        'V': ["101","101","101","101","010"],
        'Y': ["101","101","010","010","010"],
        'G': ["011","100","101","101","011"],
        'M': ["101","111","101","101","101"],
        'K': ["101","101","110","101","101"],
        'X': ["101","101","010","101","101"],
        '0': ["111","101","101","101","111"],
        '1': ["010","110","010","010","111"],
        '2': ["111","001","111","100","111"],
        '3': ["111","001","111","001","111"],
        '4': ["101","101","111","001","001"],
        '5': ["111","100","111","001","111"],
        '6': ["111","100","111","101","111"],
        '7': ["111","001","010","010","010"],
        '8': ["111","101","111","101","111"],
        '9': ["111","101","111","001","111"],
    }
    return F


def draw_text(img: np.ndarray, x: int, y: int, text: str, 
              color: Tuple[int, int, int] = (0, 0, 0), scale: int = 2):
    """Draw text on an image using 3x5 pixel font."""
    F = _font_3x5()
    cx = x
    H, W, _ = img.shape
    for ch in text.upper():
        pat = F.get(ch, F[' '])
        for r, row in enumerate(pat):
            for c, bit in enumerate(row):
                if bit == '1':
                    yy = y + r*scale
                    xx = cx + c*scale
                    if 0 <= yy < H and 0 <= xx < W:
                        img[yy:yy+scale, xx:xx+scale, :] = color
        cx += (3*scale + scale)  # glyph width + 1 space


def render_grid_image(grid: np.ndarray, scale: int = 16, 
                      palette: Dict[int, Tuple[int, int, int]] = PALETTE) -> np.ndarray:
    """Render a grid as a scaled RGB image."""
    rgb = grid_to_rgb(grid, palette)
    return upsample(rgb, scale)


def render_single_cell_image(color: int, scale: int = 64,
                            palette: Dict[int, Tuple[int, int, int]] = PALETTE) -> np.ndarray:
    """Render a single color cell as a scaled RGB image."""
    grid = np.array([[color]], dtype=int)
    rgb = grid_to_rgb(grid, palette)
    return upsample(rgb, scale)
