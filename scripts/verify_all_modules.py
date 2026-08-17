import py_compile
import sys

files = [
    "models/airnet_v5.py",
    "models/airnet_v4.py",
    "losses/adaptive_loss_v5.py",
    "inference/restore_v5.py",
    "utils/checkpoint_manager.py",
    "scripts/execute_v5_pipeline.py",
    "app.py"
]

print("--- COMPILING ALL AIR-NET V5 MODULES ---")
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"[OK] {f:35s} Compiled successfully")
    except Exception as e:
        print(f"[FAIL] {f:35s} Compilation error: {e}")
        sys.exit(1)

print("\n[ALL MODULES COMPILED CLEANLY WITH ZERO SYNTAX ERRORS]")
