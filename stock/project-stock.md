# 딥러닝 실전! 주식 투자에 응용하기

## 목적

주식의 저점, 고점, 시가, 종가를 통해 주식의 수익률을 예측한다.

## 진행 과정 Overview

1. 주식 데이터 구성
2. RNN을 통해 주식 데이터 학습
3. 각 종목 별 1주 뒤 주식 결과를 예측
4. 그 중 수익률이 높은 상위 10개에 대해서 총 100만원 투자
5. 결과 확인 및 총 정리
6. 일련의 과정을 모듈화하는 작업 

## 주식 데이터 구성 방법

### 고려사항

- User defined class를 통해서 주식 데이터 구성.

    `Stack Data`로 통칭한다.

- 주식 데이터는 `6개월간의 최근 데이터`를 통해서 구성된다.
- `torch.Tensor`와 호환되어야한다.
    - 이 때 텐서의 크기는 `C x N x 4`로 이루어지며,
    - C는 종목 개수,
    - N은 각 종목에 대한 데이터 개수
    - 4는 피쳐 개수로

        `Open, Close, Low, High` 4개의 피쳐로 이루어진다.

- Training data를 구성할 때

    `StackData` 안에서 구현하는게 좋을까? 아니면 바깥에 구현하는게 좋을까?

    ⇒ `StackData` 클래스 안에서 구현하는것이 좋아보임.

## Recurrent Neural Network

### 고려사항

- RNN을 사용하는 방법에는 torch.RNN을 쓰는 방법과 하나하나 implement하는 방법이 있지만, 이 중 RNN 모듈 사용이 더욱 이해하기 편리할 듯
    - 그러나 실제로 RNN을 구현하는것도 고려해봄직하다
    - 실제로 `RNN을 직접 구현`
- LSTM vs RNN?
    - Long Short-Term Memory의 특징은, 초반 입력에 대한 가중치가 사라지지 않는다는 점
    - 그러나 주식 데이터는 이전의 데이터보다 최근 데이터가 더 중요하게 작용된다
- Regularization issue
    - 거래량에 과도한 과중치가 부여되는것을 막기 위해, L2 norm을 통해 가중치에 대한 효과를 moderate
- Overfitting issue
    - 학습 데이터를 고정시키는 것이 아니라, 복원 추출을 통해 rnn을 학습
    - 가능하다면, dropout 계층을 이용하여 데이터에 노이즈 추가
- Exploding Gradient
    - 학습 과정 중 gradient가 과도하게 올라기기 때문에 제대로 된 fitting을 할 수 없는 이슈
    - [https://machinelearningmastery.com/exploding-gradients-in-neural-networks/](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)를 참고
    1. Weighting Regulation
        - 위에서 이미 고려. 이번 분석에서는 100의 가중치를 부여
    2. Redesign Neural Network
        - 이미 RNN이 simple network layer로 구성되어있음
    3. Gradient Clipping
        - Gradient의 크기를 제한시킴
        - 이를 고려해봄직함
    4. LSTM
        - LSTM을 통해 gradient explode를 저해할 수 있음

## 그러나...

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc1dbba1-31fc-4a0e-bf86-6934235c578e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc1dbba1-31fc-4a0e-bf86-6934235c578e/Untitled.png)

simple RNN model loss values for each training

Loss function의 값이 training이 지속되면서 우하향 그래프를 그려야할 것으로 기대되었으나, 실제로 데이터를 보니 단순 우하향이 아니라 특정 구간에서 다시 올라가고 내려가는 등의 변동적인 그래프가 나타남

특히, 같은 그래프라고 하더라도 특정 주식에 대한 사람들의 인지 등의 변화로 인하여 서로 다른 결과를 내포할 수 있음

불규칙적인 그래프 결과를 통해 알 수 있었던 점은 다음과 같다

1. 주식 데이터 분석에 있어서는 단순 거래량 및 지표가 아닌 정성적인 데이터가 들어갈 수 있다
2. 같은 시간 윈도우를 준다고 하더라도, 절대적인 시간에 대한 정보가 주식에 반영될 수 있다. 즉, 각 주식 데이터에 대한 coefficient가 time-dependent variable이라는 것

## 프로젝트를 마치면서

### 목적

주식 데이터 예측을 통해 직접 코드를 만짐으로서  `pytorch` 및 `DL`에 대해서 더 깊게 공부하기 위함

갖가지 딥러닝 학습을 시도할때 생기는 문제에 대해서 고찰, 따라서 현재 주류 딥러닝 학문이 어떤 것들을 보완하기 위함인지에 대한 background 지식 쌓기

직접 RNN을 심플하게나마 구현함으로서 앞으로 LSTM을 포함한 다른 논문에서 소개한 RNN 모델 구현 기초 다지기

### 맞닥뜨린 문제

1. 데이터 셋을 어떻게 처리할 것인가
    - 직접 class를 만드는 시도를 하였지만, 결과적으로는 그렇게 의미가 있지는 않았음
    - 비효율적으로 객체를 만드는 행위는 오히려 유지보수를 힘들게 할 우려가 있다는 것을 인지
2. RNN 모듈을 쓸 것인가? 만들 것인가?
    - RNN 모듈을 사용하게 되면 cell 안의 구체적인 information flow를 변경하기 힘듦
    - 따라서 RNN을 직접 nn.Module에서 상속받아 구현
    - 이 과정에서, single_train_data는 (t x 1 x n)의 dim을 가지고 있어야한다는 것을 인지
3. fitting 중간 nan 값이 생성됨
    - 값이 infinity에 가깝가 된다면, 계산 중간에 결측치로 인지
    - 이유는 exploding gradient가 나타났기 때문
        - 많은 경우 RNN은 classification에서 쓰이기 때문에 vanishing gradient에 많이 치중
        - Regression에서는 결과 값이 bounded되어있지 않기 때문에 exploding이 일어날 여지가 큼
    - Learning Rate를 1e-15정도로 굉장히 적게 부여함
        - 이를 통해 local minimum을 더 잘 찾기 위함
        - 다만, learning rate가 낮을 수록 학습시간이 더딘 문제가 있음
    - Regularization을 100정도로 부여함
        - 이를 통해 parameter의 가중치를 낮게 유지할 수 있음
        - 다만, 너무 regularization coefficient가 높게 잡히면 더 이상 cost function을 신경쓰지 않게 됨

    이를 통해 gradient explode 문제를 해결할 수 있었음

### 결과 및 후기

Cost function이 더 이상 특정 value 이하로 낮아지지 않았기에, 주식 데이터 자체로는 예측이 불가함을 인지함

실제로 얻은 예측 모델 및 결과가 전무하지만, 그 과정에서 RNN을 implement하는 등의 학습 용도로는 적합했음

다음 예측 모델 생성 프로젝트에서도 비슷하게 pytorch를 사용 가능할 것으로 예상됨