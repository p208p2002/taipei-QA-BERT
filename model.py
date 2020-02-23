def use_model(model_name, config_file_path, model_file_path, vocab_file_path):
    if(model_name == 'bert'):
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
        model_config, model_class, model_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
        config = model_config.from_pretrained(config_file_path)
        model = model_class.from_pretrained(model_file_path, from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
        tokenizer = model_tokenizer(vocab_file=vocab_file_path)
        return model, tokenizer
    elif(model_name == 'albert'):
        from transformers import AdamW
        from albert.albert_zh import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification
        model_config, model_class, model_tokenizer = (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
        config = model_config.from_pretrained(config_file_path,num_labels = 149)
        model = model_class.from_pretrained(model_file_path, config=config)
        tokenizer = model_tokenizer.from_pretrained(vocab_file_path)
        return model, tokenizer
