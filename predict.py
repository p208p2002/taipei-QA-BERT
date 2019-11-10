from transformers import BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    q_input = '臺灣新文化運動紀念館開館及參觀時間'
    bert_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q_input)))
    assert len(bert_ids) <= 512
    print(bert_ids)
