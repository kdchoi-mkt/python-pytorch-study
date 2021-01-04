# Recommendation System
여러 Collaborative Filtering을 pyTorch를 통해 구현합니다.

# Preliminary

## Recommendation System
+ 추천 시스템
+ 유저들에게 특정 상품 및 아이템을 추천해주는 것을 의미
+ 일반적으로 `item-based`와 `personalized-based`로 나뉨
+ `personalized-based`의 경우에는 `collaborative filtering` method를 이용하여 RS를 구축
### Explicit Data
+ 데이터를 통해서 사람들이 상품 또는 아이템을 선호하는지 안하는지 알 수 있는 데이터
+ Ex) 네이버 영화 평점, 상품 평점 등의 점수 데이터

### Implicit Data
+ 데이터 자체로 사람들이 상품 또는 아이템을 선호하는지 알 수 없지만, 적어도 사람들이 그 아이템을 선호할 때 존재하는 데이터
+ 예를 들어, 사람들이 관심이 있는 상품에 대해서는 클릭을 하거나 구매를 하지만
+ 100번 클릭한 것과 1번 클릭한 것에는 선호 정도의 차이가 있을 뿐 1번 클릭을 비선호한다고 볼 수 없음
+ Ex) 클릭 데이터, 구매 데이터 등의 비점수화 데이터

## Collaborative Filtering
+ 협력적 필터링
+ 비슷한 사용자들의 선호도를 통해 새로운 사용자의 선호도를 예측하여 높은 순으로 아이템을 보여주는 것
+ collaborative filtering 문제를 해결하는데에는 여러가지 메소드가 있으며
+ 가장 흔하게 사용되는 것은 `matrix factorization method`임

## Matrix Factorization
+ 유저 - 아이템 행렬 $R$을 생각해보자
+ $R$의 원소(entry) $r_{ui}$는 유저 u가 아이템 i에 대해 선호하는 정도(또는 평점)를 나타냄
    + 특히, $R$은 sparce matrix로, 대부분의 값이 0인 행렬임
    + 이는 모두 미관측치이며, matrix factorization은 이러한 missing value를 채우는 기법임
+ $R=U \times I$로 쪼갤 수 있다면, U와 I를 역산하여 missing value를 채울 수 있음
+ 이를 통해 $\hat{R}$을 정의할 수 있으며, $\hat{r}_{ui}$가 예측치임

## Alternating Least Squares
+ 일반적으로 matrix factorization을 하는 것은 optimization method가 들어갈 수 밖에 없음
+ 보편적으로 쓰이는 경사 하강법(Gradient Descent)을 사용하는 경우에는 time complexity가 non-polynomial
+ 따라서, Alternating Least Squares 기법을 사용하여 시간을 단축시킴

### How to use
+ U를 고정시키고 I를 최적화한다
+ I를 고정시키고 U를 최적화한다
+ 위를 계속 반복한다

## Performance Measurement
+ Explicit과 Implicit의 경우 각각에 대해서 measurement가 다름
### Mean Square Error
+ Explicit 데이터에서만 사용할 수 있는 measurement

### Precision@K
+ 각 사람별 예측 기준 TOP K 중 실제로 좋아하는 TOP K 안에 드는 데이터 비중

### Recall@K
+ 각 사람별 실제로 좋아하는 TOP K 중 예측 기준 TOP K 안에 드는 데이터 비중

### APR (Average Relevant Position)
+ 작성중...