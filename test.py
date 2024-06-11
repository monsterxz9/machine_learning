import torch
def test_gpu_computation():
    # 定义向量大小
    N = 1500000000
    # 在GPU上创建两个大小为N的张量
    a = torch.ones(N, device='cuda')
    input(" ")
if __name__ == "__main__":
    test_gpu_computation()
