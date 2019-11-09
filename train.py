from preprocess_data import convert_data_to_feature
from core import makeDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    data_feature = convert_data_to_feature()
    input_ids = data_feature['input_ids']
    input_masks = data_feature['input_masks']
    input_segment_ids = data_feature['input_segment_ids']
    answer_lables = data_feature['answer_lables']
    dataset = makeDataset(input_ids = input_ids, input_masks = input_masks, input_segment_ids = input_segment_ids, answer_lables = answer_lables)
    train_dataloader = DataLoader(dataset,batch_size=8)

