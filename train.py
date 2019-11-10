from preprocess_data import convert_data_to_feature
from core import makeDataset
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
import torch
if __name__ == "__main__":
    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    
    # setting device
    device = torch.device('cuda')

    data_feature = convert_data_to_feature()
    input_ids = data_feature['input_ids']
    input_masks = data_feature['input_masks']
    input_segment_ids = data_feature['input_segment_ids']
    answer_lables = data_feature['answer_lables']
    
    train_dataset, test_dataset = makeDataset(input_ids = input_ids, input_masks = input_masks, input_segment_ids = input_segment_ids, answer_lables = answer_lables)
    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True)

    config = bert_config.from_pretrained('bert-base-chinese',num_labels = 149)
    model = bert_class.from_pretrained('bert-base-chinese', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model.zero_grad()
    for epoch in range(5):
        
        running_loss_val = 0.0
        for batch_index, batch_dict in enumerate(train_dataloader):
            model.train()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                # attention_mask=batch_dict[1],
                labels = batch_dict[3]
                )
            loss = outputs[0]
            loss.sum().backward()
            optimizer.step()
            # scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)
            print("epoch:%2d batch:%4d train:%f"%(epoch+1, batch_index+1, running_loss_val))
        
        running_loss_val = 0.0
        for batch_index, batch_dict in enumerate(test_dataloader):
            model.eval()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                # attention_mask=batch_dict[1],
                labels = batch_dict[3]
                )
            loss = outputs[0]
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)
            print("epoch:%2d batch:%4d train:%f"%(epoch+1, batch_index+1, running_loss_val))

    
