# PyInstaller spec for the copy-to-run folder (P12). Run through build.ps1,
# not directly: build.ps1 also puts the models, settings and prompts beside
# the program, where paths.app_root() finds them.
#
# One-folder mode: dist\pyvolis\pyvolis.exe, with every library in
# dist\pyvolis\_internal\. A console program, as Rust volis is: the window
# opens from a double-click, and the same exe takes --report, --listen and
# file mode from a terminal.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

binaries = []
datas = []
hiddenimports = collect_submodules("pyvolis")

# Libraries that are loaded by path with ctypes, so the import analysis can't
# see them: llama.cpp (llama.dll, ggml*.dll, mtmd.dll; the GPU build adds
# ggml-cuda.dll) and sherpa-onnx (its extension and onnxruntime.dll 1.28.2).
for package in ("llama_cpp", "sherpa_onnx"):
    binaries += collect_dynamic_libs(package)
    datas += collect_data_files(package)
hiddenimports += collect_submodules("llama_cpp")

# librosa declares its contents in .pyi stubs that lazy_loader reads at run time.
datas += collect_data_files("librosa", includes=["**/*.pyi", "**/*.txt", "**/*.json"])
datas += collect_data_files("lazy_loader", includes=["**/*.pyi"])
# sacrebleu's chrF imports its tokenizers by name.
hiddenimports += collect_submodules("sacrebleu")

a = Analysis(
    ["pyvolis/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyInstaller"],
    # transformers and torch read their own source at run time (inspect,
    # lazy model classes), so they are kept as .py files too.
    module_collection_mode={"transformers": "pyz+py", "torch": "pyz+py", "peft": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pyvolis",
    console=True,
    upx=False,  # compressed DLLs are what security scanners flag; and torch's break
)
coll = COLLECT(exe, a.binaries, a.datas, name="pyvolis", upx=False)
