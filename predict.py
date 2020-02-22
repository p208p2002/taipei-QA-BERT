from transformers import BertTokenizer
import torch
import pickle
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW

if __name__ == "__main__":
    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    pkl_file = open('trained_model/data_features.pkl', 'rb')
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']
    
    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    config = bert_config.from_pretrained('trained_model/config.json')
    model = bert_class.from_pretrained('trained_model/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.eval()

    #
    q_inputs = ['為何路邊停車格有編號的要收費，無編號的不用收費','債權人可否向稅捐稽徵處申請查調債務人之財產、所得資料','想做大腸癌篩檢，不知如何辨理']
    for q_input in q_inputs:
        bert_ids = toBertIds(tokenizer,q_input)
        assert len(bert_ids) <= 512
        input_ids = torch.LongTensor(bert_ids).unsqueeze(0)

        # predict
        outputs = model(input_ids)
        predicts = outputs[:2]
        predicts = predicts[0]
        max_val = torch.max(predicts)
        label = (predicts == max_val).nonzero().numpy()[0][1]
        ans_label = answer_dic.to_text(label)
        
        print(q_input)
        print(ans_label)
        print()

