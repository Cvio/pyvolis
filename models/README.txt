pyvolis models. Nothing here is committed except this file; pyvolis never downloads anything.

  vad\silero_vad.onnx     voice activity detector (as Rust volis)
  asr\<name>\             recognizers: a Rust volis folder with engine.toml, or a Hugging Face
                          download (.\fetch-model.ps1 <id> -Role asr)
  tts\<name>\             Piper voices with engine.toml (as Rust volis)
  mt\<file>.gguf          the translator Rust volis also uses (exactly one at this level)
  mt\<name>\              further translators, one folder per model
                          (.\fetch-model.ps1 <id> -Role mt -Include "*Q4_K_M.gguf")

Run ".\.venv\Scripts\python.exe -m pyvolis --report" to see what was found and why anything
isn't usable.
