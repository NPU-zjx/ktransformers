"""Expose named local instrumentation RPCs alongside veRL's existing worker API."""
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
import gpu_measure


class ProfileWorker(vLLMColocateWorkerExtension):
    def gpu_reference_install(self):
        return gpu_measure.install(self)

    def gpu_reference_begin(self, folder, mode):
        return gpu_measure.begin(self, folder, mode)

    def gpu_reference_finish(self):
        return gpu_measure.finish(self)
