class TrainResult:
    def __init__(self, nmae: float = float('NaN'), mape: float = float('NaN'),
                 rmse: float = float('NaN'),
                 mae: float = float('NaN')):

        self.nmae = nmae
        self.mape = mape
        self.rmse = rmse
        self.mae = mae
