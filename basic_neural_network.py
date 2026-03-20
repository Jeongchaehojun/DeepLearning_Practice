# 원본에 제가 주석을 추가했습니다.

# # 파이토치로 구현하는 신경망
# ## 신경망 모델 구현하기

#딥러닝은 sklearn을 안쓸 수 있지만 일단 이건 기본적인 ann이라서 이 머신러닝 라이브러리 사용
import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# make_blobs()함수는 예제용 데이터셋을 만들어주는 함수이다.
n_dim = 2
# 데이터와 레이블
x_train, y_train = make_blobs(n_samples=80, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)
# 학습이 끝난 신경망 성능 평가
x_test, y_test = make_blobs(n_samples=20, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)

#label_map() 함수의 역할: 이번 신경망은 두 가지 레이블만 예측하는 아주 기본 모델이라서 4개의 레이블을 2개로 축약
# 0번이나 1번이면 0번 레이블을 갖도록,
# 2번이나 3번이면 1번 레이블을 갖도록.
def label_map(y_, from_, to_):
    y = numpy.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y

y_train = label_map(y_train, [0, 1], 0)
y_train = label_map(y_train, [2, 3], 1)
y_test = label_map(y_test, [0, 1], 0)
y_test = label_map(y_test, [2, 3], 1)

#데이터가 제대로 만들어지고 레이블링 되었는지 확인. 맷플롯립으로 시각화. 
#레이블이 0이면 점으로, 1이면 십자가로.
def vis_data(x,y = None, c = 'r'):
    if y is None:
        y = [None] * len(x)
    for x_, y_ in zip(x,y):
        if y_ is None:
            plt.plot(x_[0], x_[1], '*',markerfacecolor='none', markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c+'o' if y_ == 0 else c+'+')

plt.figure()
vis_data(x_train, y_train, c='r')
plt.show()

##원래는 거대모델에서 학습할 때 batch로 소규모 추출을 해야하지만 이건 어차피 데이터가 적다.


#방금 생성한 넘파이 벡터 형식 데이터를 파이토치 텐서로 변경
x_train = torch.FloatTensor(x_train)
print(x_train.shape)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

#torch.nn.Module 이런 식으로 신경망 모듈을 상속받은 파이썬 클래스로 정의.
class NeuralNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(NeuralNet, self).__init__() #우리가 만든 클래스가 nn.Module 클래스의 속성들을 가지고 초기화.
            self.input_size = input_size #신경망에 입력되는 데이터의 차원
            self.hidden_size  = hidden_size 

            #torch.nn.Linear()는 행렬곱과 편향을 포함하는 연산을 지원하는 객체를 반환함.
            # linear_1, linear_2를 반환하는데 나중에 함수로 쓰일 수 있음.
            # relu, sigmoid는 말 그대로 활성화 함수.
            # linear_2를 거치면 어느 값이라도 취할 수 있지만 0과1 중 어느 카테고리에 속할 것인지 분류 불가.
            # 따라서 0과 1사이의 임의의 수로 제한 해주는 시그모이드 통과해서 0과 1 중 어디에 가까운지 알자.
          
            self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.linear_2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, input_tensor):
            linear1 = self.linear_1(input_tensor)
            relu = self.relu(linear1)
            linear2 = self.linear_2(relu)
            output = self.sigmoid(linear2)
            return output


model = NeuralNet(2, 5)
learning_rate = 0.03
criterion = torch.nn.BCELoss() #여러 오차 함수 중에: "이진 교차 엔트로피" - 틀린 정도
epochs = 2000 #전체 학습 데이터를 총 몇 번 모델에 입력할지 정하는 변수.

# 최적화 알고리즘: 확률적 경사하강법
# 새 가중치 = 가중치 - 학습률*가중치에 대한 기울기
# step()을 부를 때마다 가중치를 학습률만큼 갱신한다.
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


model.eval()
test_loss_before = criterion(model(x_test).squeeze(), y_test)
print('Before Training, test loss is {}'.format(test_loss_before.item()))


# 오차값이 0.73 이 나왔습니다. 이정도의 오차를 가진 모델은 사실상 분류하는 능력이 없다고 봐도 무방합니다.
# 자, 이제 드디어 인공신경망을 학습시켜 퍼포먼스를 향상시켜 보겠습니다.

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_output = model(x_train)
    train_loss = criterion(train_output.squeeze(), y_train) #모델의 결괏값과 레이블값의 차원을 맞추기 squeeze()
    if epoch % 100 == 0:
        print('Train loss at {} is {}'.format(epoch, train_loss.item()))

    # ⭐️오차함수를 가중치로 미분하여 오차가 최소가 되는 방향을 구하고, 그 방향으로 모델을 학습률만큼 이동시킨다.
    # 신경망학습의 핵심인 '역전파'를 수행하는 코드
    train_loss.backward()
    optimizer.step()


model.eval()
test_loss = criterion(torch.squeeze(model(x_test)), y_test)
print('After Training, test loss is {}'.format(test_loss.item()))


# 학습을 하기 전과 비교했을때 현저하게 줄어든 오차값을 확인 하실 수 있습니다.
# 지금까지 인공신경망을 구현하고 학습시켜 보았습니다.
# 이제 학습된 모델을 .pt 파일로 저장해 보겠습니다.

torch.save(model.state_dict(), './model.pt')
print('state_dict format of the model: {}'.format(model.state_dict()))


# `save()` 를 실행하고 나면 학습된 신경망의 가중치를 내포하는 model.pt 라는 파일이 생성됩니다. 아래 코드처럼 새로운 신경망 객체에 model.pt 속의 가중치값을 입력시키는 것 또한 가능합니다.

new_model = NeuralNet(2, 5)
new_model.load_state_dict(torch.load('./model.pt'))
new_model.eval()
print('벡터 [-1, 1]이 레이블 1을 가질 확률은 {}'.format(new_model(torch.FloatTensor([-1,1])).item()))

