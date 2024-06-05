import torch.nn as nn

def calc_continuity_loss(predicted_paths, true_paths):
    """Calculate continuity loss of the predicted_paths
        Args: predicted_paths: [batch, n_vocab, 32, 32]
              true_paths: [batch, 1, 32, 32]
        Returns: total_loss: float
    """
    PATH_INDEX = 1
    true_paths = true_paths.view(true_paths.size(0), true_paths.size(2), true_paths.size(3))
    predicted_paths = nn.functional.softmax(predicted_paths, dim=1)
    paths = predicted_paths[:, 1, :, :]

    vertical_diff = (paths[:, :-1, :] * paths[:, 1:, :])
    vertical_mask = true_paths[:, :-1, :]
    vertical_loss = (1 - vertical_diff) * vertical_mask

    horizontal_diff = (paths[:, :, :-1] * paths[:, :, 1:])
    horizontal_mask = true_paths[:, :, :-1]
    horizontal_loss = (1 - horizontal_diff) * horizontal_mask

    diag1_diff = (paths[:, :-1, :-1] * paths[:, 1:, 1:])
    diag1_mask = true_paths[:, :-1, :-1]
    diag1_loss = (1 - diag1_diff) * diag1_mask

    diag2_diff = (paths[:, :-1, 1:] * paths[:, 1:, :-1])
    diag2_mask = true_paths[:, :-1, 1:]
    diag2_loss = (1 - diag2_diff) * diag2_mask

    total_loss = vertical_loss.sum() + horizontal_loss.sum() + diag1_loss.sum() + diag2_loss.sum()
    total_loss /= (vertical_mask.sum() + horizontal_mask.sum() + diag1_mask.sum() + diag2_mask.sum() + 0.1e-10)
    assert(total_loss >= 0)
    assert(total_loss <= 1)
    return total_loss

def calc_obstacle_loss(predicted_paths, obstacle_map):
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
    predicted_paths = nn.functional.softmax(predicted_paths, dim=1)
    paths = predicted_paths[:, PATH_INDEX, :, :]

    # 障害物とぶつかっているセルを確認
    collision = paths * obstacle_map.squeeze(1)
    collision_loss = collision.sum()

    total_loss = collision_loss / (obstacle_map.sum() + 1e-10)
    return total_loss

def calc_start_loss(predicted_paths, start_map):
    """
    Calculate start point loss of the predicted_paths
    Args:
        predicted_paths: [batch, n_vocab, 32, 32]
        start_map: [batch, 1, 32, 32]
    Returns:
        total_loss: float
    """
    PATH_INDEX = 1
    predicted_paths = nn.functional.softmax(predicted_paths, dim=1)
    paths = predicted_paths[:, PATH_INDEX, :, :]

    # スタート地点とぶつかっているセルを確認
    collision = paths * start_map.squeeze(1)
    collision_loss = collision.sum()

    total_loss = 1 - collision_loss / (start_map.sum() + 1e-10)
    return total_loss


def calc_goal_loss(predicted_paths, goal_map):
    """
    Calculate goal point loss of the predicted_paths
    Args:
        predicted_paths: [batch, n_vocab, 32, 32]
        goal_map: [batch, 1, 32, 32]
    Returns:
        total_loss: float
    """
    PATH_INDEX = 1
    predicted_paths = nn.functional.softmax(predicted_paths, dim=1)
    paths = predicted_paths[:, PATH_INDEX, :, :]

    # ゴール地点とぶつかっているセルを確認
    collision = paths * goal_map.squeeze(1)
    collision_loss = collision.sum()

    total_loss = 1 - collision_loss / (goal_map.sum() + 1e-10)
    return total_loss


def calc_solved_rate(predicted_paths, obstacle_map):
    """
    Calculate solved rate of the predicted_paths
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
    collision_count = collision.sum(dim=[1, 2])

    no_clollision_ratio = (collision_count == 0).float()

    accuracy = no_clollision_ratio.mean()

    return accuracy


def calc_obstacle_accuracy(predicted_paths, obstacle_map):
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
    collision_count = collision.sum(dim=[1, 2])

    total_path_cells = binary_paths.sum(dim=[1, 2])
    no_collision_ratio = (total_path_cells - collision_count) / (total_path_cells + 1e-10)

    return no_collision_ratio.mean()

if __name__ == '__main__':
    # test obstacle_loss
    import torch
    predicted_paths = torch.randn(2, 6, 32, 32)
    # obstacle_map must be binary tensor
    obstacle_map = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = calc_obstacle_loss(predicted_paths, obstacle_map)
    print(loss)
    assert loss > 0

    # test obstacle_accuracy
    accuracy = calc_obstacle_accuracy(predicted_paths, obstacle_map)
    print(accuracy)

    obstacle_map = torch.zeros(2, 1, 32, 32)
    loss = calc_obstacle_loss(predicted_paths, obstacle_map)
    print(loss)
    assert loss == 0

    accuracy = calc_obstacle_accuracy(predicted_paths, obstacle_map)
    print(accuracy)
    assert (accuracy - 1 < 1e-10)

    obstacle_map = torch.ones(2, 1, 32, 32)
    loss = calc_obstacle_loss(predicted_paths, obstacle_map)
    print(loss)
    assert loss != 0

    accuracy = calc_obstacle_accuracy(predicted_paths, obstacle_map)
    print(accuracy)
    assert (accuracy < 1e-10)

    start_map = torch.zeros(2, 1, 32, 32)
    start_map[:, :, 10, 10] = 1
    predicted_paths = torch.zeros(2, 6, 32, 32)
    predicted_paths[:, 1, 10, 10] = 100
    start_loss = calc_start_loss(predicted_paths, start_map)
    print(start_loss)
    assert (start_loss < 1e-10)

    goal_map = torch.zeros(2, 1, 32, 32)
    goal_map[:, :, 20, 20] = 1
    predicted_paths = torch.zeros(2, 6, 32, 32)
    predicted_paths[:, 1, 20, 20] = 100
    goal_loss = calc_goal_loss(predicted_paths, goal_map)
    print(goal_loss)
    assert (goal_loss < 1e-10)

    predicted_paths = torch.zeros(2, 6, 32, 32)
    predicted_paths[:, 1, 1:, 10] = 100
    predicted_paths[:, 0, 0, :10] = 100
    predicted_paths[:, 0, 0, 11:] = 100
    predicted_paths[:, 2:, 0, :10] = 100
    predicted_paths[:, 2:, 0, :11] = 100
    true_paths = torch.zeros(2, 1, 32, 32)
    true_paths[:, :, 1:, 10] = 1
    continuity_loss = calc_continuity_loss(predicted_paths, true_paths)
    print(continuity_loss)

    predicted_paths[:, 1, 10:15, 10] = 0
    predicted_paths[:, 0, 10:15, 10] = 100
    predicted_paths[:, 2:, 10:15, 10] = 100
    continuity_loss2 = calc_continuity_loss(predicted_paths, true_paths)
    print(continuity_loss2)
    assert continuity_loss2 > continuity_loss