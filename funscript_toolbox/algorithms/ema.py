class EMA:

    def __init__(self, alpha = 0.05):
        self.alpha = alpha
        self.mean = 0

    def update(self, val: float) -> float:
        self.mean = ((1.0 - self.alpha) * self.mean) + (self.alpha * val)
        return self.mean
