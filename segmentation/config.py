class Config(dict):
    def __init__(self, **kwargs):
        super().__init__()

        for k in kwargs:
            setattr(self, k, kwargs[k])
