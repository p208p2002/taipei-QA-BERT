import torch
from torch.utils.data import TensorDataset
import pickle

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def toBertIds(tokenizer,q_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q_input)))

def makeDataset(input_ids, input_masks, input_segment_ids, answer_lables):
    all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
    all_input_masks = torch.tensor([input_mask for input_mask in input_masks], dtype=torch.long)
    all_input_segment_ids = torch.tensor([input_segment_id for input_segment_id in input_segment_ids], dtype=torch.long)
    all_answer_lables = torch.tensor([answer_lable for answer_lable in answer_lables], dtype=torch.long)
    
    full_dataset = TensorDataset(all_input_ids, all_input_masks, all_input_segment_ids, all_answer_lables)
    
    # 切分訓練與測試資料集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_dataset,test_dataset
    

class DataDic(object):
    def __init__(self, answers):
        self.answers = answers #全部答案(含重複)
        self.answers_norepeat = sorted(list(set(answers))) # 不重複
        self.answers_types = len(self.answers_norepeat) # 總共多少類
        self.ans_list = [] # 用於查找id或是text的list
        self._make_dic() # 製作字典
    
    def _make_dic(self):
        for index_a,a in enumerate(self.answers_norepeat):
            if a != None:
                self.ans_list.append((index_a,a))

    def to_id(self,text):
        for ans_id,ans_text in self.ans_list:
            if text == ans_text:
                return ans_id

    def to_text(self,id):
        for ans_id,ans_text in self.ans_list:
            if id == ans_id:
                return ans_text

    @property
    def types(self):
        return self.answers_types
    
    @property
    def data(self):
        return self.answers

    def __len__(self):
        return len(self.answers)

def convert_data_to_feature(tokenizer, train_data_path):
    with open(train_data_path,'r',encoding='utf-8') as f:
        data = f.read()
    qa_pairs = data.split("\n")

    questions = []
    answers = []
    for qa_pair in qa_pairs:
        qa_pair = qa_pair.split()
        try:
            a,q = qa_pair
            questions.append(q)
            answers.append(a)
        except:
            continue
    
    assert len(answers) == len(questions)
    
    ans_dic = DataDic(answers)
    question_dic = DataDic(questions)

    q_tokens = []
    max_seq_len = 0
    for q in question_dic.data:
        bert_ids = toBertIds(tokenizer,q)
        if(len(bert_ids)>max_seq_len):
            max_seq_len = len(bert_ids)
        q_tokens.append(bert_ids)
        # print(tokenizer.convert_ids_to_tokens(tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q)))))
    
    print("最長問句長度:",max_seq_len)
    assert max_seq_len <= 512 # 小於BERT-base長度限制

    # 補齊長度
    for q in q_tokens:
        while len(q)<max_seq_len:
            q.append(0)
    
    a_labels = []
    for a in ans_dic.data:
        a_labels.append(ans_dic.to_id(a))
        # print (ans_dic.to_id(a))
    
    # BERT input embedding
    answer_lables = a_labels
    input_ids = q_tokens
    input_masks = [[1]*max_seq_len for i in range(len(question_dic))]
    input_segment_ids = [[0]*max_seq_len for i in range(len(question_dic))]
    assert len(input_ids) == len(question_dic) and len(input_ids) == len(input_masks) and len(input_ids) == len(input_segment_ids)

    data_features = {'input_ids':input_ids,
                    'input_masks':input_masks,
                    'input_segment_ids':input_segment_ids,
                    'answer_lables':answer_lables,
                    'question_dic':question_dic,
                    'answer_dic':ans_dic}
    
    output = open('trained_model/data_features.pkl', 'wb')
    pickle.dump(data_features,output)
    return data_features


