from typing import List
import torch.multiprocessing as mp
import ray

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.runners.model_runner import ModelRunner
from nanovllm.runners.ray_model_runner import RayModelRunner

class ExecutorBase():
    def __init__(self, config: Config):
        self.config = config
        self._init_executor()

    def _init_executor(self):
        raise NotImplementedError

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        raise NotImplementedError

    def exit(self):
        raise NotImplementedError

class MPExexutor(ExecutorBase):
    """multi-process
    """
    def __init__(self, config: Config):
        super().__init__(config)
    
    def _init_executor(self): 
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(self.config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(self.config, 0, self.events)

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        return self.model_runner.call("run", seqs, is_prefill)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

class RayExecutor(ExecutorBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def _init_executor(self):
        if(self.config.ray_head_ip == "localhost"):
            ray.init()
        else:
            ray.init(address = self.config.ray_head_ip + ":" + str(self.config.ray_head_port))
        
        self.runners = []
        for rank in range(self.config.tensor_parallel_size):
            runner = RayModelRunner.remote(self.config, rank, self.config.tensor_parallel_size, self.config.ray_head_ip, 2333)
            self.runners.append(runner)
        assert self.runners

        self.config.num_kvcache_blocks = ray.get(self.runners[0].get_num_kv_cache_block.remote())

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        futures = [runner.run.remote(seqs, is_prefill) for runner in self.runners]
        return ray.get(futures)[0]

    def exit(self):
        for runner in self.runners:
            runner.exit.remote()
        for runner in self.runners:
            ray.kill(runner)