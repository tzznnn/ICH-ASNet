# [ICH-ASNet: Automatic Prompt-Based Segmentation for Intracranial Hemorrhage in CT Images]
![](https://github.com/tzznnn/ICH-ASNet/blob/main/image/image.png)  

### Usage
We utilize the checkpoint from the [`vit_b`](https://github.com/facebookresearch/segment-anything) version of SAM. lease download the pre-trained model and store it in `./pretrained/sam_vit_b_01ec64.pth`.

We have evaluated our method on two publicly-available datasets: [BCIHM](https://physionet.org/content/ct-ich/1.3.1/) [Instance](https://instance.grand-challenge.org/). After downloading the datasets, use `utils/preprocess.py` to save the slices in `.npy` format and access them using the information in `dataset/excel/`.

For testing, please download our pre-trained prompt generator and model [checkpoint](https://pan.baidu.com/s/1xuT_karw01wiYAxVwByR7g). Meanwhile, please download the [weights](https://github.com/microsoft/unilm/tree/master/beit3) of our multimodal feature generator.
