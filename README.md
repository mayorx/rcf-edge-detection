# rcf

variety of Richer Convolutional Features for Edge Detection (resnet101-based)

## results

I test two types of loss: last layer loss / sum up each layer's loss.
[Detail here.](https://github.com/mayorx/rcf-edge-detection/pull/2/files?utf8=%E2%9C%93&diff=split)

1. last layer only:
ODS: **0.8135** , OIS: **0.831** on BSDS500 dataset
[pretrained model, last layer only](https://drive.google.com/open?id=12JsCtpnMpnyauoceHiJoW94jQnH6MIr8)

<img src="examples/final_layer/100007-1.png" width="100" /><img src="examples/final_layer/100007-2.png" width="100" /><img src="examples/final_layer/100007-3.png" width="100" /><img src="examples/final_layer/100007-4.png" width="100" /><img src="examples/final_layer/100007-5.png" width="100" /><img src="examples/final_layer/100007-6.png" width="100" /><img src="examples/final_layer/100007-nms.png" width="100" /><img src="examples/final_layer/100007-img.jpg" width="100" />

<img src="examples/final_layer/100039-1.png" width="100" /><img src="examples/final_layer/100039-2.png" width="100" /><img src="examples/final_layer/100039-3.png" width="100" /><img src="examples/final_layer/100039-4.png" width="100" /><img src="examples/final_layer/100039-5.png" width="100" /><img src="examples/final_layer/100039-6.png" width="100" /><img src="examples/final_layer/100039-nms.png" width="100" /><img src="examples/final_layer/100039-img.jpg" width="100" />

2. all layers:
ODS: **0.796** , OIS: **0.814**  on BSDS500 dataset
[pretrained model, all layers](https://drive.google.com/open?id=1v9QFjkKtWTwPC3vOoHsy3zi9cKk8BlWN)

<img src="examples/all_layer/100007-1.png" width="100" /><img src="examples/all_layer/100007-2.png" width="100" /><img src="examples/all_layer/100007-3.png" width="100" /><img src="examples/all_layer/100007-4.png" width="100" /><img src="examples/all_layer/100007-5.png" width="100" /><img src="examples/all_layer/100007-6.png" width="100" /><img src="examples/all_layer/100007-nms.png" width="100" /><img src="examples/all_layer/100007-img.jpg" width="100" />

<img src="examples/all_layer/100039-1.png" width="100" /><img src="examples/all_layer/100039-2.png" width="100" /><img src="examples/all_layer/100039-3.png" width="100" /><img src="examples/all_layer/100039-4.png" width="100" /><img src="examples/all_layer/100039-5.png" width="100" /><img src="examples/all_layer/100039-6.png" width="100" /><img src="examples/all_layer/100039-nms.png" width="100" /><img src="examples/all_layer/100039-img.jpg" width="100" />

## requirements

* pytorch 0.4.1
* python 3.6.6
* dataset(provide by [original repo](https://github.com/yun-liu/rcf))
    * http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst
    * http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    * http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
* and other requirements... (cv2, numpy , etc.)

## usage

#### train:

* put your data in 'data/HED-BSDS_PASCAL' (or make a soft link)
* python train.py

#### test:
* python test.py

#### a simple example:
* python demo.py

#### evaluate:

it may take several hours...

* [pretrained model, last layer only](https://drive.google.com/open?id=12JsCtpnMpnyauoceHiJoW94jQnH6MIr8) or [pretrained model, all layers](https://drive.google.com/open?id=1v9QFjkKtWTwPC3vOoHsy3zi9cKk8BlWN)
* requirements:
  * matlab
  * hed [link](https://github.com/xwjabc/hed/tree/c8ed5abc4d2b6ad2862b0d61cf6184ce2cdf3cae)
* you should modify [path to your predicts](https://github.com/mayorx/rcf/blob/master/eval_edge.m#L3) and [path to ground truth (.mat)](https://github.com/mayorx/rcf/blob/master/eval_edge.m#L39)
  * path to your predicts should contain two folders: png and mat
* sh eval.sh

## related projects

* [rcf pytorch implementation](https://github.com/meteorshowers/RCF-pytorch)
* [rcf](https://github.com/yun-liu/rcf)
* [hed pytorch implementation](https://github.com/xwjabc/hed)
