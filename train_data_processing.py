from torch.utils.data import (
    TensorDataset, DataLoader, RandomSampler
)


def train_process(batch_size, X_train, mask_train, y_train):
    train_data = TensorDataset(X_train, mask_train, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader
