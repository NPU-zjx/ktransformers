"""Install only the optional timing hooks in SGLang's Python workers."""
import os

if os.environ.get("KT_PHASE_HOOKS") == "1":
    try:
        if os.environ.get("KT_TORCH_DIAGNOSTIC") == "1":
            from torch_hooks import install
        else:
            from phase_hooks import install
        install()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(79)
