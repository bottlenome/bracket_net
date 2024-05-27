import torch.nn as nn


def obstacle_loss(predicted_paths, obstacle_map):
    """
    Calculate obstacle collision loss of the predicted_paths
    Args: 
        predicted_paths: [batch, n_vocab, 32, 32]
        true_paths: [batch, 1, 32, 32]
        obstacle_map: [batch, 1, 32, 32]
    Returns: 
        total_loss: float
    """
    PATH_INDEX = 1
    LAMBDA = 0.001
    predicted_paths = nn.functional.softmax(predicted_paths, dim=1)
    paths = predicted_paths[:, PATH_INDEX, :, :]

    # 障害物とぶつかっているセルを確認
    collision = paths * obstacle_map.squeeze(1)
    collision_loss = collision.sum()

    total_loss = collision_loss / (obstacle_map.sum() + 1e-10)
    return total_loss * LAMBDA


def obstacle_accuracy(predicted_paths, obstacle_map):
    """
    Calculate obstacle collision accuracy of the predicted_paths
    Args: 
        predicted_paths: [batch, n_vocab, 32, 32]
        obstacle_map: [batch, 1, 32, 32]
    Returns: 
        accuracy: float
    """
    PATH_INDEX = 1
    predicted_paths = nn.functional.softmax(predicted_paths, dim=1)
    binary_paths = (predicted_paths.argmax(dim=PATH_INDEX) == 1).float()

    # 障害物とぶつかっているセルを確認
    collision = binary_paths * obstacle_map.squeeze(1)
    collision_count = collision.sum()

    # 障害物とぶつかっていないセルの割合を計算
    total_path_cells = binary_paths.sum()
    no_collision_ratio = (total_path_cells - collision_count) / total_path_cells

    return no_collision_ratio

if __name__ == '__main__':
    # test obstacle_loss
    import torch
    predicted_paths = torch.randn(2, 6, 32, 32)
    # obstacle_map must be binary tensor
    obstacle_map = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = obstacle_loss(predicted_paths, obstacle_map)
    print(loss)
    assert loss > 0

    # test obstacle_accuracy
    accuracy = obstacle_accuracy(predicted_paths, obstacle_map)
    print(accuracy)

    obstacle_map = torch.zeros(2, 1, 32, 32)
    loss = obstacle_loss(predicted_paths, obstacle_map)
    print(loss)
    assert loss == 0

    accuracy = obstacle_accuracy(predicted_paths, obstacle_map)
    print(accuracy)
    assert (accuracy - 1 < 1e-10)

    obstacle_map = torch.ones(2, 1, 32, 32)
    loss = obstacle_loss(predicted_paths, obstacle_map)
    print(loss)
    assert loss != 0

    accuracy = obstacle_accuracy(predicted_paths, obstacle_map)
    print(accuracy)
    assert (accuracy < 1e-10)