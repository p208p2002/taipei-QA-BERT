# 台北QA問答機器人(with BERT or ALBERT)
問一個問題，告訴你應該去哪個單位處理這些問題

## 檔案說明
- train.py : 模型訓練(BERT fine-tune)
- predict.py : 提問預測
- [taipei-qa-bert.pdf](https://github.com/p208p2002/taipei-QA-BERT/blob/master/taipei-qa-bert.pdf) : 投影片檔案

## 中文Albert
- 預設使用bert-base-chinese
- 欲切換至albert-zh-tiny請將`train.py`與`predict.py`對應註解拿掉
    > 使用albert-zh-tiny需要對訓練參數進行微調，否則無法獲得好的效果
- 更多albert-zh模型與用法請參閱[p208p2002/albert-zh-for-pytorch-transformers](https://github.com/p208p2002/albert-zh-for-pytorch-transformers)

## 環境需求
- python 3.6+
- pytorch 1.3+
