def break_text(train):
    sentences = train.text.values
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = train.target.values
    return sentences, labels
