class SDSubseedParams:
    def __init__(self,
                 subseed=-1,
                 subseed_strength=0,
                 seed_resize_from_h=-1,
                 seed_resize_from_w=-1):
        self.seed: int = subseed
        self.strength: float = subseed_strength
        self.resize_from_h: int = seed_resize_from_h
        self.resize_from_w: int = seed_resize_from_w