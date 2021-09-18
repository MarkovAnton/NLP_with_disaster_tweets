import numpy as np
from transformers import Trainer


def prediction(custom_model, test_dataset):
    test_trainer = Trainer(custom_model)
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    y_pred = np.argmax(raw_pred, axis=1)

    return y_pred
