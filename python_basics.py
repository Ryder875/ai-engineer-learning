import math


def mean_squared_error(predictions, targets):
    """计算两个等长、非空列表之间的均方误差。"""

    # zip 遇到较短列表会停止，因此先检查长度。
    if len(predictions) != len(targets):
        raise ValueError("预测值和真实值的长度必须相同")

    if len(targets) == 0:
        raise ValueError("输入列表不能为空")

    total_squared_error = 0.0

    for prediction, target in zip(predictions, targets):
        error = prediction - target
        total_squared_error += error ** 2

    # 全部累加完成后，再求平均并返回。
    mse = total_squared_error / len(targets)
    return mse


# 示例一：三个误差都是 1，所以 MSE 为 1。
result_1 = mean_squared_error(
    [2.0, 4.0, 6.0],
    [1.0, 3.0, 5.0],
)
print("示例一 MSE:", result_1)

# 示例二：误差为 0、2、-1，所以 MSE 为 5/3。
result_2 = mean_squared_error(
    [1.0, 5.0, 4.0],
    [1.0, 3.0, 5.0],
)
print("示例二 MSE:", result_2)


# 基本测试：浮点数使用 isclose 判断是否接近。
assert math.isclose(result_1, 1.0)
assert math.isclose(result_2, 5.0 / 3.0)

# 预测完全正确时，MSE 应为零。
assert mean_squared_error(
    [1.0, 3.0, 5.0],
    [1.0, 3.0, 5.0],
) == 0.0

print("MSE 基本测试通过")