import torch
import pickle
from core import toBertIds

if __name__ == "__main__":    
    # load and init
    pkl_file = open('trained_model/data_features.pkl', 'rb')
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']
        
    # BERT
    # from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
    # model_config, model_class, model_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    # config = model_config.from_pretrained('trained_model/config.json')
    # model = model_class.from_pretrained('trained_model/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    # tokenizer = model_tokenizer(vocab_file='bert-base-chinese-vocab.txt')
    
    # ALBERT
    from transformers import AdamW
    from albert.albert_zh import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification 
    model_config, model_class, model_tokenizer = (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
    config = model_config.from_pretrained('trained_model/config.json')
    model = model_class.from_pretrained('trained_model/pytorch_model.bin', config=config)
    tokenizer = model_tokenizer.from_pretrained('albert/albert_tiny/vocab.txt')
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

