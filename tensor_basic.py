# 텐서는 파이토치에서 다양한 수식을 계산하는 데 사용하는 가장 기본적인 자료구조.
"""
1 -> 스칼라, 모양은 []
[1,2,3] -> 벡터, 모양은 [3]
[[1,2,3]] -> 행렬, 모양은[1,3]
[[[1,2,3]]] -> n랭크 텐서, 모양은 [1,1,3]
"""
# unsqueeze(), squeeze(), view()
# # 3.1 텐서와 Autograd
# ## 3.1.1 텐서 다루기 기본:  차원(Rank)과 Shape

import torch

x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(x)
print("Size:", x.size())
print("Shape:", x.shape)
print("랭크(차원):", x.ndimension())


# 랭크 늘리기
x = torch.unsqueeze(x, 0)
print(x)
print("Size:", x.size())
print("Shape:", x.shape)
print("랭크(차원):", x.ndimension())


# 랭크 줄이기
x = torch.squeeze(x)
print(x)
print("Size:", x.size())
print("Shape:", x.shape) #[3, 3] 2개의 차원에 각 3개의 원소를 가진 텐서
print("랭크(차원):", x.ndimension())


# 랭크의 형태 바꾸기
x = x.view(9)
print(x)
print("Size:", x.size())
print("Shape:", x.shape)
print("랭크(차원):", x.ndimension())


try:
    x = x.view(2,4)
except Exception as e:
    print(e) #에러 출력

