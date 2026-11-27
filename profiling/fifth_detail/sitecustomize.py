"""External hooks only. Preserve the native publication/phase sitecustomize."""
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import runpy
import sys

if os.environ.get('KTRL_DETAIL_PROFILE'):
    here = Path(__file__).resolve()
    for entry in sys.path:
        candidate = Path(entry)/'sitecustomize.py'
        if candidate.is_file() and candidate.resolve() != here:
            runpy.run_path(str(candidate))
            break
    from worker_hooks import patch

    class Loader(importlib.abc.Loader):
        def __init__(self, base): self.base = base
        def create_module(self, spec):
            return self.base.create_module(spec) if hasattr(self.base, 'create_module') else None
        def exec_module(self, module):
            self.base.exec_module(module)
            patch(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            names = {'transformers.trainer'} if os.environ['KTRL_DETAIL_ROLE'] == 'actor' else {
                'sglang.srt.managers.tp_worker', 'sglang.srt.managers.scheduler'}
            if name in names:
                spec = None
                # Delegate through existing native hook finders, including KT_PHASE.
                # Calling PathFinder directly would suppress their loader wrappers.
                for finder in tuple(sys.meta_path):
                    if finder is self or not hasattr(finder, 'find_spec'): continue
                    spec = finder.find_spec(name, path, target)
                    if spec is not None: break
                if spec:
                    spec.loader = Loader(spec.loader)
                return spec

    sys.meta_path.insert(0, Finder())
