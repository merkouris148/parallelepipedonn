class DeltaSchedule:
    def __init__(
            self,
            delta_origin: float,
            delta_final: float,
            max_it: int
        ) -> None:
    
        self.delta_origin   = delta_origin
        self.delta_final    = delta_final
        self.max_it         = max_it

        self.delta_current  = delta_origin
        self.reduce_rate    = (self.delta_origin - self.delta_final) / self.max_it
    

    def current(self) -> float:
        return self.delta_current
    
    
    def update(self) -> None:
        if self.delta_current > self.delta_final:
            self.delta_current -= self.reduce_rate