import torch
def test_pos_weight_shape():
    # pos_weight的长度必须等于标签数K
    assert pos_weight.shape[0] == K

def test_loss_no_nan_extreme_logits():
    # 用极端logits（例如±100）测试BCEWithLogitsLoss不产生NaN
    extreme_logits = torch.tensor([[100.0, -100.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 1.0]])
    loss = loss_fn(extreme_logits, targets)
    assert not torch.isnan(loss).any()
    loss.backward()
    assert not torch.isnan(extreme_logits.grad).any()

def test_pos_weight_direction():
    # 低频标签的pos_weight必须大于高频标签
    assert pos_weight[low_freq_idx] > pos_weight[high_freq_idx]
