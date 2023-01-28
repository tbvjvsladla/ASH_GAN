---
layout: post
title:  "SRCNN 설계-코드연습"
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


## SRCNN 네트워크를 코드레벨에서 한번 공부해보자



```python
import torch.nn as nn

class SRCNN_tutoral(nn.Module):
    #네트워크 첫번째 : __init__ 메서드랑 forward메서드 이거 두개 있는게 기본구조

    def __init__(self, num_channels = 1): #초기 이미지 입력은 num_channels라고 부른다
        #여기서 num_channel = 1이면 그레이 스케일, 3이면 RGB이다
        #pass
        super(SRCNN_tutoral).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        #매개변수 :입력채널, 출력채널, 커널사이즈, 패딩
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(128, 2, kernel_size=5, padding=5//2)
        #논문에서 최종 출력 채널이 L, ab니까 2 인것임
        self.relu = nn.ReLU(inplace=True)

        #레이어 설계는 끝, 이제 순방향, 역방향 알려주는게 forward
    
    def forward(self, input_layer): #input_layer은 통상 x로 쓴다
        #pass
        conv1 = self.conv1(input_layer)
        conv1 = self.relu()
        conv2 = self.conv2(conv1)
        conv2 = self.relu()
        conv3 = self.conv3(conv2)

        return conv3
    #위 forward를 통상 편하게는 output를 다 x로 쓴다
        
```

설계한 SRCNN 네트워크 구조의 이미지는 아래와 같다   
![img_SRCNN]({{site.url}}/images/images_cha.png)
채널이 조금 안맞는거 같지만 튜토리얼은 넘어가자...   
그림상 self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=5//2)   
128이 아니라 32 인거같음


편하게 쓴 forward 메서드

```python
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu()
        x = self.conv2(x)
        x = self.relu()
        x = self.conv3(x)

        return x
```


