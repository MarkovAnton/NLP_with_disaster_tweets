from transformers import BertTokenizer
from tqdm.auto import tqdm


def tokenization_sentence(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_sentences = [tokenizer.tokenize(sent) for sent in tqdm(sentences)]
    return tokenized_sentences, tokenizer
