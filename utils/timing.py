import time
import torch
if torch.cuda.is_available():
    import torch.cuda as cuda
import signal

def handle_timeout(signum, frame):
    raise TimeoutError(f"Timer::TimeoutError")

class Timer:
    def __init__(self, device:torch.device = torch.device("cpu")) -> None:
        self.start_time:dict = {}
        self.result:dict = {}
        self.sync = lambda: None
        if device.type == "cuda":
            self.sync = cuda.synchronize

    def start(self, name:str='timer') -> None:
        if name in self.result:
            del self.result[name]
        self.sync()
        self.start_time[name] = time.time()

    def stop(self, name:str='timer') -> float:
        self.sync()
        end_time = time.time()
        self.result[name] = (end_time - self.start_time[name])
        return self.result[name]
    
    def timeout(self, timeout:int, handler=handle_timeout) -> bool:
        # handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))
        return
