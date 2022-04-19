## [HQGA](https://arxiv.org/pdf/2112.06197.pdf): Video as Conditional Graph Hierarchy for Multi-Granular Question Answering

![teaser](https://github.com/doc-doc/HQGA/blob/main/introduction.png)

## Todo
2. [x] Release features of [NExT-QA](https://drive.google.com/file/d/1vU9IEW0GvXz3wzumfu9X8lm4ri2SBjLB/view?usp=sharing)([BERT feature](https://drive.google.com/file/d/1KtpduE0SViUYFjZq81hlSmgEr7e2QrUa/view?usp=sharing) are from [NExT-QA](https://github.com/doc-doc/NExT-QA))[2021/12/23].

## Environment

Anaconda 4.8.4, python 3.6.8, pytorch 1.6 and cuda 10.2. For other libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python==3.6.8
>conda activate videoqa
>git clone https://github.com/doc-doc/HQGA.git
>pip install -r requirements.txt
```
## Data Preparation
We use MSVD-QA as an example to help get farmiliar with the code. Please download the pre-computed features and trained models [here](https://drive.google.com/file/d/1MrupFq8jubEA4nEl4CppR5Rddz9rW_6Z/view?usp=sharing)

After downloading the data, please create a folder ```['data/']``` at the same directory as ```['HQGA']```, then unzip the video and QA features into it. You will have directories like ```['data/msvd/' and 'HQGA/']``` in your workspace. Please move the model file ```[.ckpt]``` into ```['HQGA/models/msvd/']```. 


## Usage
Once the data is ready, you can easily run the code. First, to test the environment and code, we provide the prediction and model of the HQGA on MSVD-QA. 
You can get the results reported in the paper by running: 
```
>python eval_oe.py
```
The command above will load the prediction file under ['results/msvd/'] and evaluate it. 
You can also obtain the prediction by running: 
```
>./main.sh 0 test #Test the model with GPU id 0
```
The command above will load the model under ['models/msvd/'] and generate the prediction file.
If you want to train the model (Please follow our paper for details.), please run
```
>./main.sh 0 train # Train the model with GPU id 0
```
It will train the model and save to ['models/msvd']. 

## Result Visualization
![vis-res](https://github.com/doc-doc/HQGA/blob/main/vis-res.png)
**Example from NExT-QA dataset.
## Citation
```
@proceedings{xiao2021video,
      title={Video as Conditional Graph Hierarchy for Multi-Granular Question Answering}, 
      author={Junbin Xiao and Angela Yao and Zhiyuan Liu and Yicong Li and Wei Ji and Tat-Seng Chua},
      booktitle={Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI)},
      year={2022},
}
```
## Acknowledgement
Our feature extraction for object, frame appearance and motion are from [BUTD](https://github.com/MILVLG/bottom-up-attention.pytorch) and [HCRN](https://github.com/thaolmk54/hcrn-videoqa) respectively. Many thanks the authors for their great work and code!

## Modification for NeXT-QA dataset

1. main_qa.py 
```
dataset = 'nextqa'  
multi_choice = True
model_prefix = 'bert-16c20b-2L05GCN-FCV-AC-ZJ-6c5s'

```

2.videoqa.py
```
num_clip, num_frame, num_bbox = 16, 16*4, 20  # For nextqa

```

3.sample_loader.py
```
self.max_qa_length = 20  # 20 for MSRVTT, MSVD, TGIF-QA Trans & Action, 37 for nextqa
self.bbox_num = 10  # 20 for NExT-QA, 10 for others

def get_multi_choice_sample(self, idx):

        temporal_multihot = self.get_tce_and_tse(qns)

```

4.EncoderQns.py
```
self.max_qa_length = 37  # Same in sample_loader.py
self.temporal_length = 11  # total number of category and signal in get_tce_and_tse() in sample_loader.py
```

5.EncoderVid.py
```
75 framePos = framePos.contiguous().view(batch_size, num_clip, frame_pclip, region_pframe, -1)

```
