---
layout: post
title:  "GAN 수업 - 설계한 모델 객체화&옵티마이저, main구문 작성"
categories: coding
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# GAN코드 수업 : 마지막 코드 구동부 수업 리뷰



여기서는 로스함수까지 설계한 GAN코드에 대한 객체화 및 딥러닝 훈련을 수행한다.



```python
from psutil import virtual_memory
import torch

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)

if gpu_info.find('falied') >= 0:
    print("GPU연결 실패")
else:
    print("GPU구동 성공")

ram_gb = virtual_memory().total / 1e9
print('{:.1f} gigabytes of available RAM\n'.format(ram_gb))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('학습을 진행하는 기기:',device)
```

<pre>
GPU구동 성공
68.6 gigabytes of available RAM

학습을 진행하는 기기: cuda
</pre>

```python
from fastai.data.external import untar_data, URLs
import glob

coco_path = untar_data(URLs.COCO_SAMPLE)
paths = glob.glob(str(coco_path) + "/train_sample/*.jpg")

import numpy as np
import time

np.random.seed(seed = int(time.time())) #시간으로 시드 입력
chosen_paths = np.random.choice(paths, 20000, replace=False)
index = np.random.permutation(20000) #이미지 인덱스 랜덤으로 섞기

train_path = chosen_paths[index[:15000]] #0~15000 장을 훈련 데이터로 사용
val_paths = chosen_paths[index[15000:]] #15001~마지막 까지는 검증데이터로 사용
```


```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image   #PIL 라이브러리는 Opencv와 동일한 기능이나, FastAi가 PIL예제가 많다
from skimage.color import rgb2lab, lab2rgb  #논문에서 RGB -> Lab 수행
import numpy as np
```

### 이미지를 DataLoadar에 넣기 위해서 전처리 작업을 하는 클래스 설계



```python
class ColorizationDataset(Dataset) : #논문의 이미지 전처리 클래스
    def __init__(self, paths, mode='train'):
        self.mode = mode
        self.paths = paths

        if mode == 'train': #256x256리사이즈, 이미지 인터폴레이션 BICUBIC, 어그멘테이션 수행
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.RandomHorizontalFlip()
            ])
        elif mode == 'val': #256x256리사이즈, 이미지 인터폴레이션 BICUBIC
            self.transforms = transforms.Resize((256,256), Image.BICUBIC)
        else:
            raise Exception("모드입력 에러")


    def __getitem__(self, index): #이미지 불러오고 RGB -> Lab수행, 텐서형식 변환
        img = Image.open(self.paths[index]).convert("RGB")
        img = np.array(self.transforms(img))
        img = rgb2lab(img).astype("float32")
        img = transforms.ToTensor()(img)

        #노멀라이제이션 수행
        L = img[[0], ...]
        ab = img[[1,2], ...]
        L = L / 50. -1
        ab = ab / 110.

        return {'L':L, 'ab':ab}
        
    
    def __len__(self):
        return len(self.paths)
```


```python
#훈련데이터와 검증데이터를 DataLoader에 넣기 위해 설계한 ColrizationDataset 전처리 클래스 객체화
dataset_train = ColorizationDataset(train_path, mode='train')
dataset_val = ColorizationDataset(val_paths, mode='val')
```

<pre>
C:\Users\HILS_AMD\AppData\Local\Temp\ipykernel_2276\4148482005.py:8: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  transforms.Resize((256, 256), Image.BICUBIC),
c:\Users\HILS_AMD\AppData\Local\Programs\Python\Python310\lib\site-packages\torchvision\transforms\transforms.py:329: UserWarning: Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
  warnings.warn(
C:\Users\HILS_AMD\AppData\Local\Temp\ipykernel_2276\4148482005.py:12: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  self.transforms = transforms.Resize((256,256), Image.BICUBIC)
</pre>

```python
dataloader_train = DataLoader(dataset_train, batch_size = 16, num_workers=0, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size = 16, num_workers=0, pin_memory=True)

#Dataloader에 입력한 훈련,검증데이터의 구조를 출력해보자
data = next(iter(dataloader_train))
Ls, abs = data['L'], data['ab']
print(Ls.shape, abs.shape)
#출력 결과는 [batch_size, Channel개수, tr리사이즈 크기(256,256)]이 나올것이다.
```

<pre>
torch.Size([16, 1, 256, 256]) torch.Size([16, 2, 256, 256])
</pre>
### 논문의 생성기(Generator)과 판별기(Discriminator) 클래스 설계



```python
#이제 Pix2Pix_Generator과 Pix2Pix_Discriminator설계를 수행한다.
import torch.nn as nn

#U-Net 구조의 Pix2Pix_Generator
class Pix2Pix_Generator(nn.Module):
    def __init__(self):
        super(Pix2Pix_Generator, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.encoder_1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.encoder_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.encoder_3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.encoder_4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.encoder_5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.encoder_6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.middle = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.decoder_6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5) #probability of an element to be zeroed. Default: 0.5
        )

        self.decoder_5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        self.decoder_4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        self.decoder_3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5)
        )

        self.decoder_2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5)
        )

        self.decoder_1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5)
        )

        self.output_layer = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Hyperbolic Tangent
        )

    def forward(self, x):
        input_layer = self.input_layer(x)
        encoder_1 = self.encoder_1(input_layer)
        encoder_2 = self.encoder_2(encoder_1)
        encoder_3 = self.encoder_3(encoder_2)
        encoder_4 = self.encoder_4(encoder_3)
        encoder_5 = self.encoder_5(encoder_4)
        encoder_6 = self.encoder_6(encoder_5)

        middle = self.middle(encoder_6)

        cat_6 = torch.cat((middle, encoder_6), dim=1)
        decoder_6 = self.decoder_6(cat_6)
        cat_5 = torch.cat((decoder_6, encoder_5), dim=1)
        decoder_5 = self.decoder_5(cat_5)
        cat_4 = torch.cat((decoder_5, encoder_4), dim=1)
        decoder_4 = self.decoder_4(cat_4)
        cat_3 = torch.cat((decoder_4, encoder_3), dim=1)
        decoder_3 = self.decoder_3(cat_3)
        cat_2 = torch.cat((decoder_3, encoder_2), dim=1)
        decoder_2 = self.decoder_2(cat_2)
        cat_1 = torch.cat((decoder_2, encoder_1), dim=1)
        decoder_1 = self.decoder_1(cat_1)

        output = self.output_layer(decoder_1)
        
        return output
```


```python
Pix2Pix_Generator()
```

<pre>
Pix2Pix_Generator(
  (input_layer): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
  (encoder_1): Sequential(
    (0): LeakyReLU(negative_slope=0.2, inplace=True)
    (1): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (encoder_2): Sequential(
    (0): LeakyReLU(negative_slope=0.2, inplace=True)
    (1): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (encoder_3): Sequential(
    (0): LeakyReLU(negative_slope=0.2, inplace=True)
    (1): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (encoder_4): Sequential(
    (0): LeakyReLU(negative_slope=0.2, inplace=True)
    (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (encoder_5): Sequential(
    (0): LeakyReLU(negative_slope=0.2, inplace=True)
    (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (encoder_6): Sequential(
    (0): LeakyReLU(negative_slope=0.2, inplace=True)
    (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (middle): Sequential(
    (0): LeakyReLU(negative_slope=0.2, inplace=True)
    (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (decoder_6): Sequential(
    (0): ReLU(inplace=True)
    (1): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
  )
  (decoder_5): Sequential(
    (0): ReLU(inplace=True)
    (1): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
  )
  (decoder_4): Sequential(
    (0): ReLU(inplace=True)
    (1): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
  )
  (decoder_3): Sequential(
    (0): ReLU(inplace=True)
    (1): ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
  )
  (decoder_2): Sequential(
    (0): ReLU(inplace=True)
    (1): ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
  )
  (decoder_1): Sequential(
    (0): ReLU(inplace=True)
    (1): ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
  )
  (output_layer): Sequential(
    (0): ReLU(inplace=True)
    (1): ConvTranspose2d(64, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (2): Tanh()
  )
)
</pre>

```python
# Pix2Pix_Discriminator의 구조는 C64-C128-C256-C512
class Pix2Pix_Discriminator(nn.Module):
    def __init__(self):
        super(Pix2Pix_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.model(x)
```


```python
Pix2Pix_Discriminator()
```

<pre>
Pix2Pix_Discriminator(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
)
</pre>
### 논문의 모델 초기화 부분 수행

-> 해당 부분을 수행하지 않으면 설계한 딥러닝 모델의 레이어 내 모듈이   

Local Mininum에 빠져 학습 성능이 나빠지기에   

적합한 모델 초기화 방식을 매칭하여, 학습 수행시 GM(Global Mininum)으로 수렴하게 만들자



```python
# Pix2Pix 초기화 -> 논문 6.2절을 보기
# Gaussian distribution 사용, 평균 0, 표준편차 0.02

def ini_weight(m): #m -> Layer을 줄여쓴 연구자들끼리의 약속
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        # 이 레이어의 weight의 '값'을 초기화 시키겠다는 말임
        # 초기화 함수는 각각 초기화에 필요한 매개변수가 있는데
        # normal_ 함수는 평균이랑, 표준편차를 이용해서 초기화한다.
        print("Conv2D model init!")
    
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        print("ConvTranspose2D model init!")
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        # batchnorm은 1 주변에서 정규화 하는 모델 -> 그래서 평균도 1로 놓음
        nn.init.constant_(m.bias.data, 0.)
        # 하여 바이어스가 1인 것을 특정 val(0.) 값으로 채우는
        # init.constant 초기화 함수를 같이 사용
        print("BatchNorm2D Initialized")

        #함수에 _로 끝나는 경우 : 값을 덮어씌우겠다는 뜻
        #_로 안끝나면 따로이 값을 선언하겠다는 뜻

def model_init(model): 
    # 이거는 nn.Sequential하면서 초기화 같이 하는 코드도 있고 
    # 여러 방식이 존재함
    model.apply(ini_weight)
    return model
```

### 로스 함수 설계

아래 로스 함수는 BCD에 대한 로스 함수를 사용하기 위한 전처리 클래스 설계 로 이해하면 되며,  

전처리 항목은 true/False에 대한 라벨이 붙은 Tensor행렬을 만들어서 넣는 것이다.   

L1은 생성기에서만 사용하니 L1로스 함수는 나중에 따로 사용하며,  

생성기, 판별기 모두 True -> Ture / Ture -> False / False -> True / False -> False   

에 대한 Loss를 계산해야 하기에 편의를 위해 라벨을 붙인 텐서 전처리기를 설계한다 보면 된다.



```python
class Pix2Pix_LossFunc(nn.Module):
    def __init__(self):
        super(Pix2Pix_LossFunc, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        # real_label이라는 이름을 갖는 버퍼를 생성하고,
        # 그 버퍼는 1.0이 기록된 텐서형 변수를 레지스터를 등록함.
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.BCEWithLogitsLoss() #이게 loss func함수임(클래스)..
        #시그모이드도 붙어있는 BCE 손실함수

    def get_labels(self, prediction, target_is_real): #prediction : 예측한 값
        if target_is_real:
            labels = self.real_label #라벨링을 진짜 이미지라고 박는것임
        else:
            labels = self.fake_label
        return labels.expand_as(prediction) #여기에 예측한 값(위에서 1.0, 0.0이 넣어진다 보면됨?)
        #받아온 데이터와 라벨링한 텐서값의 크기를 맞추기 위해 차원확장을 수행

    def __call__(self, prediction, target_is_real):
        labels = self.get_labels(prediction, target_is_real)
        loss = self.loss(prediction, labels)

        return loss
```


```python
#여기는 학습한 결과물을 중간에 확인하기 위한 함수임

def lab_to_rgb(L, ab):
    #학습한거 데모하려면 정규화 했던거를 원복해야함
    L = (L + 1) * 50.
    ab = ab * 110.
    #Tensor 리스트 순번도 원복하고 Numpy행려로 만들어줘야함
    Lab = torch.cat([L, ab], dim = 1).permute(0, 2, 3, 1).cpu().numpy()
    #행렬 원복은 현재 행렬이 [배치사이즈, 채널, 256, 256] 이니
    #원래의 인풋 이미지 행렬인 [배치사이즈, 256, 256, 채널]로 바꿔준다
    #원래 이미지의 행렬 순서는 -> skimage.color 라이브러리 확인
    #그리고 딥러닝 연산중인 데이터가 GPU에 있으니 이를 CPU로 전달한다.
    rgb_images = [] #이미지를 저장할 리스트 선언

    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_images.append(img_rgb)
    
    return np.stack(rgb_images, axis = 0)
```

## 이제 앞에서 쭉 설계한 클래스들에 대한 객체화를 진행

객체화 후 모델 초기화, 훈련 시 step, 방향성 설정을 위한 옵티마이저 도 만든다   



![img_optimizer]({{site.url}}/images/optimizer.png)     



그리고 `epoch`를 설정하는데 이게 딥러닝 훈련 횟수이다.   

-> 적으면 GM을 거의 못찻고, 많으면 오버피팅남



이제부터 main함수 작성(딥러닝 구동 함수 작성)

main함수 할 일



1. 설계한 클래스를 객체화 해야한다.   

2. 생성한 객체는 딥러닝 네트워크이니 -> 모델 초기화 해줘야 한다.   

3. 초기화 한 모델은 GPU로 보내줘야 한다.   

이때, GPU로 모델을 할당하는 방식은 두가지 `.to(device)`와 `.cuda()`가 있으나   

전자가 후자보다 코드 구현상 좀 더 유연하다. -> 단 전자는 개발자가 이상하게 돌아가고 있는 점을   

알아차리기는 어려울 것이다.   

![img_to_gpu]({{site.url}}/images/to_GPU.png)  

4. 옵티마이저로 선언한 생성기, 판별기의 최적화 스텝 설정해주기

5. 훈련 몇번 돌릴지 (epoch) 선언 후 구동



```python
import torch.optim as optim
from tqdm import tqdm
#Tqdm라이브러리는 모델 러닝을 수행할 때 진척도 보기 위한 라이브러리임

#생성자 1, 2, 3 수행
exam_gen = model_init(Pix2Pix_Generator())
exam_gen.to(device) #exam_gen.cuda()와 같은기능

#판별자 1, 2, 3 수행
exam_dis = model_init(Pix2Pix_Discriminator())
exam_dis.to(device)

#loss 함수는 2번 항목은 안한다 -> Loss함수는 '모델'이 아니니까
exam_bce = Pix2Pix_LossFunc().to(device)
#참고로 Loss함수 선언할때 변수명으로 'criterion'을 자주쓴다
exam_L1 = nn.L1Loss().to(device) #앞에있는 'exam_bce'는 L1이 빠졋으니 따로 GPU로 보냄

#이제 optimizer을 선언 및 설정한다
#옵티마이저의 러닝 레이트(한 스텝 단위)는 2e-4로 했다(논문이랑 다름)
#옵티마이저는 ADAM옵티마이저를 쓴다.
#가장 기본적인 옵티마이저인 SGD는 러닝레이트 하나만 매개변수로 필요하나,
#ADAM은 betas라는 매개변수도 필요하다(가우시안 함수?)
exam_opt_gen = optim.Adam(exam_gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
exam_opt_dis = optim.Adam(exam_dis.parameters(), lr=2e-4, betas=(0.5, 0.999))


epoch = 1000 #딥러닝의 최적화 돌리는 횟수 -> 이게 딥러닝 훈련 횟수라 보면됨.
```

<pre>
Conv2D model init!
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
ConvTranspose2D model init!
BatchNorm2D Initialized
ConvTranspose2D model init!
BatchNorm2D Initialized
ConvTranspose2D model init!
BatchNorm2D Initialized
ConvTranspose2D model init!
BatchNorm2D Initialized
ConvTranspose2D model init!
BatchNorm2D Initialized
ConvTranspose2D model init!
BatchNorm2D Initialized
ConvTranspose2D model init!
BatchNorm2D Initialized
ConvTranspose2D model init!
Conv2D model init!
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
BatchNorm2D Initialized
Conv2D model init!
</pre>
## 학습의 for문에서 할 일

1. 앞에 GPU에 입력한 이미지들(학습 이미지)를 생성기와, 판별기에 입력시킨다.

2. 입력 후 훈련을 시키는데 생성기 다 학습 -> 판별기 다 학습 -> 생성기 다 학습 이런식이 되어야함   

이거를 `freezing`이라 부른다.   

![img_freeze]({{site.url}}/images/model_freeze.png)    

훈련 진행 도중 zero_gard()는 해줘야 한다 -> 이거는 파이토치쓰면 항상 해야함 그냥 외우자..

![img_freeze]({{site.url}}/images/optm_zero.png)    

3. 생성기로 만든 가짜 이미지를 판별기로 보내준다. -> 예측값 생성   

이 예측값 만드는 과정에서 생성기에서 만든 이미지를 판별기로 넣어야 하는데   

생성기 이미지 -> 복사(detach()) -> 판별기 입력 -> 예측값 생성   

순으로 해야한다.   

![img_detach]({{site.url}}/images/copy_detach.png)    

4. 생성한 판별기의 예측값을 loss함수를 통해 반복학습

5. 판별기의 경우 두번 학습해야한다

   + 3, 4 과정을 통한 가짜 이미지에 대한 학습

   + 진짜 이미지에 대한 학습





Freezing구문 중   

`requires_grad = True` : 텐서를 생성하기 위해 사용하는 함수들의 파라미터로 true를 넘겨줌 -> Autograd활성화

`requires_grad_(True)` : 이미 생성된 Tensor의 멤버함수인 requires_grad_를 이용해 Autogard를 활성화

큰.. 차이는 없다.





backward()   

![img_backward]({{site.url}}/images/backward.png)   

역전파는 추측한 값에서 발생한 오류 에 비례니까 Loss함수 수행 후 나온 파라미터에 대해 역전파를 하는거임




```python
import matplotlib
import matplotlib.pyplot as plt

import warnings
#모델을 훈련시킬때마다 warning메세지가 올라오는데
#그거를 없애주기 위해 사용한 라이브러리


for e in range(epoch):
    for index, data in enumerate(tqdm(dataloader_train)):
        #원래 기본은 for index, data in enumerate(dataloader_train):
        #이중 for문인 이유는 epoch돌고, index, data를 로더에서 가져오는 순이다.

        L = data['L'].to(device) #훈련데이터 리턴값 L채널을 다시 GPU로
        ab = data['ab'].to(device)

        #위 훈련데이터 중, L채널만 생성기로 보내서 가짜 이미지 ab채널 만듬
        fake_ab_channel = exam_gen(L)

        #판별기 훈련 시작!!
        exam_dis.train()

        #훈련 순서는 판별기 -> freeze -> 생성기 -> freeze 순
        #아래 for 구문이 freezing구문임
        for parm in exam_dis.parameters():
            parm.requires_grad_(True)
        
        exam_opt_dis.zero_grad()

        #생성기에서 만든 ab채널이랑 L채널 합쳐서 가짜 이미지 만듬
        fake_image = torch.cat([L, fake_ab_channel], dim=1)

        #생성기에서 만든 이미지를 detach()함수로 `복사`해서 판별기에 넣고
        #이거로 판별기의 판별정보(`예측`)값 출력
        fake_prediction = exam_dis(fake_image.detach())

        #출력한 판별정보를 Loss 함수에 입력해서 '판별성능 결과' 출력
        cal_dis_fake_res = exam_bce(fake_prediction, False)

        #이제 진짜 이미지에 대해서 위 과정 동일하게 수행
        real_image = torch.cat([L, ab], dim=1)
        real_prediction = exam_dis(real_image.detach())
        cal_dis_real_res = exam_bce(real_prediction, True)

        cal_dis_total_res = (cal_dis_fake_res + cal_dis_real_res) * 0.5

        cal_dis_total_res.backward()

        #역전파 한 다음에 옵티마이저 불러와서 다음 스텝으로 넘어간다
        exam_opt_dis.step()



        #생성기 훈련 시작!
        exam_gen.train()

        #freezing 및 파이토치 zero_gard()
        for parm in exam_gen.parameters():
            parm.requires_grad = False
        
        exam_opt_gen.zero_grad()


        #절차지향적인 관점으로 앞에서 했던거 반복
        fake_image = torch.cat([L, fake_ab_channel], dim=1)
        fake_prediction = exam_dis(fake_image.detach())

        cal_gen_fake_bce_res = exam_bce(fake_prediction, True)
        #생성기는 판별기를 속이도록 훈련시켜야 하기에 true를 넣는다

        cal_gen_fake_L1_res = exam_L1(fake_ab_channel, ab) * 100
        #여기서 100 이 하이퍼 파라미터 람다 임

        #이게 생성기의 로스 함수 수식임
        cal_gen_fake_res = cal_gen_fake_bce_res + cal_gen_fake_L1_res

        cal_gen_fake_res.backward()
        exam_opt_gen.step()



        
```

<pre>
 17%|█▋        | 162/938 [00:59<04:32,  2.84it/s]
</pre>