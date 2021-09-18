from reading_files import read_files
from break_into_sentences import break_text
from tokenization_of_sentences import tokenization_sentence
from plotting_token_lens import plot_token_lens
from padding_sentences import pad_sentence
from train_data_processing import train_process
from val_or_test_data_processing import val_or_test_process
from training import train_data
from checking_on_test_sample import check_on_test_sample
from data_processing import *
from prediction_for_test import prediction
from writing_to_file import write_to_file

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch
import nltk
from transformers import AdamW, BertForSequenceClassification
import seaborn as sns

sns.set(font_scale=1.2)
sns.set_style(style='whitegrid')
device_num = 0


def main():
    nltk.download('punkt')
    nltk.download("stopwords")
    device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
    print(', '.join(nltk.corpus.stopwords.words('english')))

    read_files('data/train.csv', 'data/test.csv')
    train = read_files('data/train.csv', 'data/test.csv')[0]
    test = read_files('data/train.csv', 'data/test.csv')[1]
    print(train)
    print(train.info())

    sentences = break_text(train)[0]
    labels = break_text(train)[1]

    tokenized_sentences = tokenization_sentence(sentences)[0]
    tokenizer = tokenization_sentence(sentences)[1]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenized_sentences)]
    token_lens = [len(sent) for sent in tqdm(input_ids)]

    plot_token_lens(token_lens)
    print(tokenizer.convert_tokens_to_ids("[PAD]"))

    input_ids = pad_sentence(input_ids)

    get_attention_masks = lambda input_ids: [[float(i > 0) for i in seq] for seq in input_ids]
    attention_masks = get_attention_masks(input_ids)

    X_train, X_test, mask_train, mask_test, y_train, y_test = train_test_split(input_ids,
                                                                               attention_masks,
                                                                               labels,
                                                                               test_size=0.3
                                                                               )

    X_train, X_val, mask_train, mask_val, y_train, y_val = train_test_split(X_train,
                                                                            mask_train,
                                                                            y_train,
                                                                            test_size=0.1
                                                                            )

    X_train = torch.tensor(X_train)
    X_val = torch.tensor(X_val)
    X_test = torch.tensor(X_test)

    mask_train = torch.tensor(mask_train)
    mask_val = torch.tensor(mask_val)
    mask_test = torch.tensor(mask_test)

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    batch_size = 16

    train_dataloader = train_process(batch_size, X_train, mask_train, y_train)

    val_dataloader = val_or_test_process(batch_size, X_val, mask_val, y_val)

    test_dataloader = val_or_test_process(batch_size, X_test, mask_test, y_test)

    custom_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    custom_model = custom_model.to(device)

    optimizer = AdamW(custom_model.parameters(), lr=2e-5)

    custom_model, history = train_data(
        custom_model, device, optimizer,
        train_dataloader, val_dataloader,
        num_epochs=25
    )

    custom_model.eval()

    check_on_test_sample(test_dataloader, device, custom_model)

    X_test = list(test['text'])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    test_dataset = Dataset(X_test_tokenized)

    y_pred = prediction(custom_model, test_dataset)

    write_to_file(test, y_pred)


if __name__ == '__main__':
    main()
