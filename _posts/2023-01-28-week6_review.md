---
layout: post
title:  "GAN 수업 - Loss Func 클래스로 설계"
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


### 6주차 코드리뷰 - 모델 초기화 복습 및 Loss Function



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
#훈련데이터와 검증데이터를 DataLoader에 넣기 위해 설계한 ColrizationDataset 전처리 클래스를 활용
dataset_train = ColorizationDataset(train_path, mode='train')
dataset_val = ColorizationDataset(val_paths, mode='val')
```

<pre>
C:\Users\HILS_AMD\AppData\Local\Temp\ipykernel_1380\4148482005.py:8: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  transforms.Resize((256, 256), Image.BICUBIC),
c:\Users\HILS_AMD\AppData\Local\Programs\Python\Python310\lib\site-packages\torchvision\transforms\transforms.py:329: UserWarning: Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
  warnings.warn(
C:\Users\HILS_AMD\AppData\Local\Temp\ipykernel_1380\4148482005.py:12: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
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

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
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
    (2): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
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
### 모델 초기화 복습



모델초기화를 하는 이유 -> LM(Local Mininum)에 빠지는 것을 방지하기 위해   

-> 만약 빠지게 되면 학습 결과물이 매우 나쁜 결과물이 나온다   

-> 수식을 다 기억할 필요는 없으며, 사용한 레이어 모델 별로 적합한 모델 초기화 방식이 매칭되어 있다.   

+ ex ) ReLU 활성화 함수에는 He(kaiming) Initialization 방법이 가장 적합하고,   

tanh의 경우 Xavier Intialization 방법이 적합하다.   

BN 레이어의 경우 weight는 1, bias는 0으로 초기화하는게 일반적이다.   




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

### 모델의 손실함수(Loss Function)설계 공부

모델이 학습할 때 나아갈 방향성을 제시함   

 + ex) CNN평가할 때 IoU로 평가하는데 이 IoU를 미분 가능하게 만들어서    

CNN의 Loss func으로 사용한다   



Loss function의 유사어로는 Cost Func, Objective Func가 존재함   

-> 부르는 방식이 다른 이유는 각각의 최적화 방법론이 살짝 달라서?   

 + Loss Func : 하나의 input에 대한 오차를 계산

 + Cost Func : 모든 input에 대한 오차를 계산

 + Objective Func : 위 두함수는 global mininum point를 계산하는데   

Objective Func는 GMP같은 목표지점이 있을 뿐 그 목표지점이 GMP는 아니다.   





어쨋든 Loss Func는 미분가능하고 최적화 기법 써야하고 이 최적화 방식이 step단위로   

움직이기에 이 step에 대한 lr(Learning rate)을 정의해줘야 한다   



Loss Func로 주로쓰이는 대표적인 평가기법   

L1 = MAE(Mean Absolute Error), L2 = MSE(Mean Squrared Error), RMSE   



Pix2pix는 L1 로스 함수랑 BCE(Binary Cross Entropy)를 섞어서 쓴다

-> BCE는 이진 분류 할때 주로 씀 -> GAN은 Fake / Real만을 분류하기에...

-> CCE(Categorical Cross-Entropy)는 멀티 클래스 분류할때 주로쓴다.



pix2pix논문의 Loss func는 `생성기`, `판별기`가 `BCE` 붙고,   

다시 `판별기`에 `L1` 붙여서 Loss func의 Gradient decent(GD)를 진행한다.   

그리고 L1에 대한 영향력이 정의된 Hyperpram = 람다를 곱해서 더한다.   



이때, BCE의 `생성기`는 0에 가깝게, `판별기`는 1에 가깝게....   


```python

#Loss Func에 대한 클래스 설계하기

class Pix2Pix_LossFunc(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        pass



    def get_labels(self, preds, target_is_real):

        #생성기가 -> 판별기한테 이미지를 던져줌 : 1번 인풋

        #진짜 이미지 -> 판별기한테 이미지를 던져줌 : 2번 인풋

        #위 1번, 2번 인풋에 대한 구분을 위한 라벨붙이는 함수

        pass



    def __call__(self, preds, target_is_real):

        #클래스가 호출될때마다 수행해야 하는 내용 기재

        pass

```

위 클래스가 Loss Func의 메서드 함수 구조라 보면 된다.


아래에 작성한 Loss function 코드 리뷰를 수행한다.



### 잠깐! 그전에!

사실 Loss function및 optimizer은 코드 몇줄로 끝나는 짧은 부분에 속한다.   



<span style="color:red">일반 BCE Loss Function설계시 코드</span>   

![img_loss1]({{site.url}}/images/loss_1.png)   



<span style="color:red">Pix2pix 수행을 위한 BCE + L1 로스함수 코드</span>   

![img_loss1]({{site.url}}/images/loss_2.png)   

그러나, 이를 이해하기 쉽게 하려고 loss function을 클래스로 설계하고 있으며,   

optimizer과 함께 수행됨을 좀 나누어서 설명을 진행한다.





+ 첫번째로 `register_buffer` : 진짜 텐서, 가짜 텐서 만들기 -> 그냥 torch.tensor로 만들면 된다.   

하지만 만든 레이어를 `기록`하고 싶다 -> 모델을 저장하고 싶다.   

저장은 되고, 역전파 할때 변경은 안되게 하는 방법!    

self.register_buffer을 사용함   

이것에 대한 다른 블로그의 설명은 아래와 같다.   

  + optimizer가 업데이트하지 않는다.

  + 그러나 값은 존재한다(하나의 layer로써 작용한다고 보면 된다.)

  + state_dict()로 확인이 가능하다.

  + GPU연산이 가능하다.    



따라서 네트워크를 구성함에 있어서   

네트워크를 end2end로 학습시키고 싶은데 중간에 업데이트를 하지않는   

일반 layer를 넣고 싶을 때 사용할 수 있다.



+ expand() 메소드는 원하는 차원 크기를 input으로 받아    

텐서의 값들을 뒤쪽 axis에서 반복하여   

확장된 차원의 반복 텐서를 생성한다.   

이때 expend_as() expend() 함수와 동일한 기능을 하나,   

input으로 shape가 아닌 tensor가 직접 들어가는 점이 다르다.   



+ BCEWithLogitsLoss는 클래스이고, 최종적으로 사용할때는 `input`와 `target`값 두개가 필요하다.   

![img_loss_parm]({{site.url}}/images/loss_input.png)  

어쨋든 정답, 예측 두개 값을 받아야 한다. 




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
        return labels.expend_as(prediction) #여기에 예측한 값(위에서 1.0, 0.0이 넣어진다 보면됨?)
        #받아온 데이터와 라벨링한 텐서값의 크기를 맞추기 위해 차원확장을 수행

    def __call__(self, prediction, target_is_real):
        labels = self.get_labels(prediction, target_is_real)
        loss = self.loss(prediction, labels)
        # 사용한 loss Func가 예측값으로는 모델 연산에서 수행한 Prediction
        # 정답값으로는 laber -> Tensor로 1.0, 0.0으로 값을 밀어넣었으니
        # 이 부분은 강사님께 한번 확언을 듣자!
        return loss
        
```


```python
```
