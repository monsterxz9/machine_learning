import torch
def test_gpu_computation():
    # 定义向量大小
    N = 600000000
    # 在GPU上创建两个大小为N的张量
    a = torch.ones(N, device='cuda')
    b = torch.ones(N, device='cuda')
    # 在GPU上进行向量加法运算
    c = a + b
    # 在GPU上验证结果是否正确
    if torch.all(c == 2):
        print("GPU computation test PASSED")
    else:
        print("GPU computation test FAILED")
if __name__ == "__main__":
    test_gpu_computation()
