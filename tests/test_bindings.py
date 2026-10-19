"""ABI checks do not need a GPU or the vendor library."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rs_depth._cuda import CFrameMetadata, CStreamMetadata


def test_ctypes_layout_matches_compiled_header(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler required for native layout check")
    source = tmp_path / "layout.c"
    source.write_text('''
#include <stddef.h>
#include <stdio.h>
#include "rs_depth_abi.h"
int main(void) {
    printf("%zu %zu %zu %zu %zu %zu %zu\\n",
           sizeof(StreamMetadata), sizeof(FrameMetadata),
           offsetof(FrameMetadata, depth), offsetof(FrameMetadata, color),
           offsetof(FrameMetadata, ir_format), offsetof(FrameMetadata, width),
           offsetof(FrameMetadata, height));
    return 0;
}
''')
    exe = tmp_path / "layout"
    subprocess.run([cc, "-I", str(ROOT / "cuda"), str(source), "-o", str(exe)],
                   check=True, capture_output=True, text=True)
    native = [int(v) for v in subprocess.check_output([str(exe)], text=True).split()]
    python = [ctypes.sizeof(CStreamMetadata), ctypes.sizeof(CFrameMetadata),
              CFrameMetadata.depth.offset, CFrameMetadata.color.offset,
              CFrameMetadata.ir_format.offset, CFrameMetadata.width.offset,
              CFrameMetadata.height.offset]
    assert python == native
