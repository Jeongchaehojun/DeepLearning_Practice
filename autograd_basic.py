#Autograd는 미분계산을 자동화하여 경사 하강법을 구현하는 수고를 덜어줌

import torch

#w의 requres_grad를 True로 설정하면 파이토치의 autograd 기능이 계산할 때 w에 대한 미분 값을 자동으로 w.grad에 저장한다.
w = torch.tensor(1.0, requires_grad=True)


a = w*3
l = a**2
# l = a^2 = (3w)^2 = 9 w^2

# 연쇄법칙을 활용함. a, w를 차례대로 미분
l.backward()
print(w.grad)
print('l을 w로 미분한 값은 {}'.format(w.grad))

