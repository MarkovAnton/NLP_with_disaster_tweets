import matplotlib.pyplot as plt
import seaborn as sns


def plot_token_lens(token_lens):
    plt.figure(figsize=(12, 7))

    sns.displot(token_lens, kde=False)
    plt.xlim([0, 1024])
    plt.xlabel('Длина предложения')
    plt.ylabel('Количество предложений')
    plt.title('Распределение длин предложений')

    plt.show()
