from core import convert_data_to_feature, make_dataset, split_dataset, compute_accuracy, use_model
from torch.utils.data import DataLoader
import torch
from transformers import AdamW

if __name__ == "__main__":    
    # BERT
    model_setting = {
        "model_name":"bert", 
        "config_file_path":"bert-base-chinese", 
        "model_file_path":"bert-base-chinese", 
        "vocab_file_path":"bert-base-chinese-vocab.txt",
        "num_labels":149  # 分幾類 
    }    

    # # ALBERT
    # model_setting = {
    #     "model_name":"albert", 
    #     "config_file_path":"albert/albert_tiny/config.json", 
    #     "model_file_path":"albert/albert_tiny/pytorch_model.bin", 
    #     "vocab_file_path":"albert/albert_tiny/vocab.txt",
    #     "num_labels":149 # 分幾類
    # }    

    #
    model, tokenizer = use_model(**model_setting)
    
    # setting device    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if(device =='cuda' and torch.cuda.device_count()>1):
        model = torch.nn.DataParallel(model,device_ids=[0,1])
    print("using device",device)
    model.to(device)

    #    
    data_feature = convert_data_to_feature(tokenizer,'Taipei_QA_new.txt')
    input_ids = data_feature['input_ids']
    input_masks = data_feature['input_masks']
    input_segment_ids = data_feature['input_segment_ids']
    answer_lables = data_feature['answer_lables']
    
    #
    full_dataset = make_dataset(input_ids = input_ids, input_masks = input_masks, input_segment_ids = input_segment_ids, answer_lables = answer_lables)
    train_dataset, test_dataset = split_dataset(full_dataset, 0.9)
    train_dataloader = DataLoader(train_dataset,batch_size=24,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=8,shuffle=True)    

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model.zero_grad()
    for epoch in range(30):
        running_loss_val = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(train_dataloader):
            model.train()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            # batch_dict = tuple(t.to('cuda:1') for t in batch_dict)
            outputs = model(
                batch_dict[0],
                # attention_mask=batch_dict[1],
                labels = batch_dict[3]
                )
            loss,logits = outputs[:2]
            if(device =='cuda' and device,torch.cuda.device_count()>1):
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            # scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            
            # compute the loss
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(logits, batch_dict[3])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # log
            print("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
        
        running_loss_val = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(test_dataloader):
            model.eval()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                # attention_mask=batch_dict[1],
                labels = batch_dict[3]
                )
            loss,logits = outputs[:2]
            if(device =='cuda' and device,torch.cuda.device_count()>1):
                loss = loss.mean()
            
            # compute the loss
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(logits, batch_dict[3])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # log
            print("epoch:%2d batch:%4d test_loss:%2.4f test_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained('trained_model')
