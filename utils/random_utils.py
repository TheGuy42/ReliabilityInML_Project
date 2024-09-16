# from __future__ import annotations
import torch



class Generators:
    generator = torch.Generator()
    generator.manual_seed(0)
    def __init__(self):    
        pass

    @staticmethod
    def rand_seed() -> int:
        return torch.randint(high=2**32, size=(1,), generator=Generators.generator).item()

    @staticmethod 
    def new() -> torch.Generator:
        new_gen = torch.Generator()
        new_gen.manual_seed(Generators.rand_seed())
        return new_gen


class RandGenerator:
    def __init__(self, seed:int=42):   
        # super().__init__()
        # self.manual_seed(seed=seed) 
        self.generator:torch.Generator = torch.Generator()
        self.generator.manual_seed(seed)

    def rand_seed(self) -> int:
        return torch.randint(high=2**32, size=(1,), generator=self.generator).item()

    def new_generator(self):
        seed = self.rand_seed()
        new_gen = torch.Generator()
        new_gen.manual_seed(seed)
        return new_gen




