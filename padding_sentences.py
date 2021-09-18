from keras.preprocessing.sequence import pad_sequences

# выберем максимальную длину предложения
MAX_LEN = 256


def pad_sentence(input_ids):
    padding = lambda texts: pad_sequences(texts,
                                          maxlen=MAX_LEN,
                                          dtype="long",
                                          truncating="post",
                                          padding="post"
                                          )

    # применяем padding и truncation ко входным данным
    input_ids = padding(input_ids)

    return input_ids
