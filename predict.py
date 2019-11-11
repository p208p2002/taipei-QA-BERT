from transformers import BertTokenizer
import torch
import pickle
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW

if __name__ == "__main__":
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    q_input = '臺灣新文化運動紀念館開館及參觀時間'
    bert_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q_input)))
    assert len(bert_ids) <= 512
    print(bert_ids)
    input_ids = torch.LongTensor(bert_ids).unsqueeze(0)

    pkl_file = open('trained_model/data_features.pkl', 'rb')
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']

    # load and init model
    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    config = bert_config.from_pretrained('trained_model/config.json')
    model = bert_class.from_pretrained('trained_model/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.eval()

    # predict
    outputs = model(input_ids)
    predicts = outputs[:2]
    predicts = predicts[0]
    max_val = torch.max(predicts)
    label = (predicts == max_val).nonzero().numpy()[0][1]
    ans_label = answer_dic.to_text(label)
    
    print(q_input)
    print(ans_label)

    # print(predict[0].shape)
