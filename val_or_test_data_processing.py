from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler
)


def val_or_test_process(batch_size, X, mask, y):
    data = TensorDataset(X, mask, y)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader
