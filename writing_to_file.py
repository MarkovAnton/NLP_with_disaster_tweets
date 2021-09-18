def write_to_file(test, y_pred):
    test['target'] = y_pred
    test.drop(['text'], axis=1, inplace=True)
    print(test)

    test.to_csv('submission.csv', index=False)
