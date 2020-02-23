from albert_zh import AlbertConfig,AlbertForSequenceClassification,AlbertTokenizer
import torch
if __name__ == "__main__":
    tokenizer = AlbertTokenizer.from_pretrained('albert_tiny/vocab.txt')
    model_config = AlbertConfig.from_json_file('./albert_tiny/config.json')
    model = AlbertForSequenceClassification.from_pretrained('./albert_tiny',config = model_config)

    intput_str = '周杰倫，臺灣著名華語流行歌曲男歌手、音樂家、唱片製片人。同時是演員、導演，也是電競團隊隊長兼老闆、服飾品牌老闆。以其個人風格和聲歌手樂創作能力著稱，影響華語樂壇。 在2000年，周杰倫發行了他的首張專輯《Jay》，從屬於唱片公司阿爾發音樂。'
    input_ids = torch.tensor(tokenizer.encode(intput_str)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]
    
    print(loss,logits)