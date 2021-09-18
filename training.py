from collections import defaultdict
from tqdm.auto import tqdm
import time

import numpy as np
import torch
from IPython.display import clear_output
from plotting_learning_curves import plot_learning_curves


def train_data(
        model,
        device,
        optimizer,
        train_batch_gen,
        val_batch_gen,
        num_epochs=50
):
    # Функция для обучения модели и вывода лосса и метрики во время обучения.

    history = defaultdict(lambda: defaultdict(list))

    for epoch in tqdm(range(num_epochs), desc='epochs'):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        start_time = time.time()

        model.train(True)  # устанавливаем поведение dropout / batch_norm  в обучение

        # На каждой "эпохе" делаем полный проход по данным
        for X_batch, mask_batch, y_batch in tqdm(
                train_batch_gen, desc='train sample batches'
        ):
            # Обучаемся на батче (одна "итерация" обучения нейросети)
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            model_output = model(X_batch, mask_batch, labels=y_batch)
            loss = model_output.loss
            logits = model_output.logits

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            train_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # Подсчитываем лоссы и сохраням в "историю"
        train_loss /= len(train_batch_gen)
        train_acc /= len(train_batch_gen)
        history['loss']['train'].append(train_loss)
        history['acc']['train'].append(train_acc)

        model.train(False)  # устанавливаем поведение dropout / batch_norm  в тестирование

        # Полный проход по валидации
        for X_batch, mask_batch, y_batch in tqdm(
                val_batch_gen, desc='validation sample batches'
        ):
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            # При использовании .no_grad() модель не будет считать и хранить градиенты.
            # Это ускорит процесс предсказания меток для тестовых данных.
            with torch.no_grad():
                model_output = model(X_batch, mask_batch, labels=y_batch)
                loss = model_output.loss
                logits = model_output.logits

            val_loss += np.sum(loss.detach().cpu().numpy())

            # применяем функцию max к каждому логиту из батча
            # функция max возвращает две величины: значения и индексы
            # выбираем индексы
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            val_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # Подсчитываем лоссы и сохраням в "историю"
        val_loss /= len(val_batch_gen)
        val_acc /= len(val_batch_gen)
        history['loss']['val'].append(val_loss)
        history['acc']['val'].append(val_acc)

        clear_output()

        # Печатаем результаты после каждой эпохи
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
        print("  training accuracy: \t\t\t{:.2f} %".format(train_acc * 100))
        print("  validation accuracy: \t\t\t{:.2f} %".format(val_acc * 100))

        if epoch > 0:
            plot_learning_curves(history)

    return model, history
