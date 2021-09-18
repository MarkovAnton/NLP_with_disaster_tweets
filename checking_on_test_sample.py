from tqdm.auto import tqdm
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score


def check_on_test_sample(test_dataloader, device, custom_model):
    test_preds, test_labels = [], []

    for X_batch, mask_batch, y_batch in tqdm(test_dataloader):
        X_batch = X_batch.to(device)
        mask_batch = mask_batch.to(device)

        # При использовании .no_grad() модель не будет считать и хранить градиенты.
        # Это ускорит процесс предсказания меток для тестовых данных.
        with torch.no_grad():
            logits = custom_model(X_batch, mask_batch).logits

        # применяем функцию max к каждому логиту из батча
        # функция max возвращает две величины: значения и индексы
        # выбираем индексы
        y_pred = logits.max(1)[1].detach().cpu().numpy()

        test_preds.extend(y_pred)
        test_labels.extend(y_batch.numpy())

        print('Accuracy: {0:.2f}%, Precision: {1:.2f}%, Recall: {2:.2f}%'.format(
            accuracy_score(test_labels, test_preds) * 100,
            precision_score(test_labels, test_preds) * 100,
            recall_score(test_labels, test_preds) * 100
        ))
