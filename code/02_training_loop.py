"""第 2 课：训练循环。先手写版，再 PyTorch 版，对比两者。

运行：python 02_training_loop.py
PyTorch 版需要 pip install torch；没装会自动跳过。
"""

areas = [50.0, 80.0, 100.0]
prices = [100.0, 160.0, 200.0]


def manual_version(lr=0.0001, steps=10):
    """手写版：自己算梯度。"""
    w = 3.0
    for step in range(steps):
        grad = 0.0
        for x, y in zip(areas, prices):
            error = w * x - y  # 预测 - 真实
            grad += 2 * error * x  # MSE 对 w 的导数
        grad /= len(areas)
        w -= lr * grad  # 和第 1 课的公式一模一样
        print(f"[手写] 第 {step + 1} 步: w = {w:.4f}")
    return w


def torch_version(lr=0.0001, steps=10):
    """PyTorch 版：框架帮你算梯度。"""
    import torch

    xs = torch.tensor(areas)
    ys = torch.tensor(prices)
    w = torch.tensor(3.0, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=lr)

    for step in range(steps):
        pred = w * xs  # 1. 预测
        loss = ((pred - ys) ** 2).mean()  # 2. 算 loss（MSE）
        optimizer.zero_grad()
        loss.backward()  # 3. 自动算梯度
        optimizer.step()  # 4. 更新 w
        print(f"[torch] 第 {step + 1} 步: w = {w.item():.4f}")
    return w.item()


if __name__ == "__main__":
    manual_version()
    try:
        torch_version()
    except ImportError:
        print("未安装 PyTorch，跳过 torch 版。可在 Google Colab 运行。")

    # 练习 A：把 lr 改成 0.0002 试试，会发生什么？
