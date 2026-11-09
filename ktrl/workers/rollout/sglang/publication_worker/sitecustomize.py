import os

if os.environ.get("KTRL_PUBLICATION_OUTPUT"):
    try:
        if os.environ.get('KTRL_REQUEST_SEEDS')=='1':
            from ktrl.workers.rollout.sglang.request_seeds import install as install_seeds
            install_seeds()
        if os.environ.get("KTRL_ONLINE_PHASES") == "1":
            from phase_hooks import install as install_phases
            install_phases()
        from ktrl.weights.serving import install
        install()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(79)
