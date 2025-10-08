from BaseNNModel import BaseNNModel

class YourModel(BaseNNModel):

    def load(self, path: str) -> None:
        # Пример на псевдо-питоновском
        # with open(path, 'rb') as inp:
        #   weights = inp.read()
        #   self._model = nn.Pypeline(
        #     nn.Layer1(weights=weights[0]),
        #     nn.Layer2(weights=weights[1]),
        #   )
        pass

    def preprocessing(self, path: str) -> object:
        # Пример на псевдо-питоновском
        # with open(path, 'rb') as inp:
        #   x = nn.Tiff2Image(inp)
        #   preproced_x = nn.normalize(x)
        # return preproced_x
        pass

    def predict(self, path: str) -> object:
        # Пример на псевдо-питоновском
        # x = self.preprocessing(path)
        # return self._model.predict(x)
        pass