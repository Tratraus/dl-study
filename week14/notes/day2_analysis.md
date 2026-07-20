Day2 完成记录:
- pos_weight计算公式: w_i = (N - n_pos_i) / n_pos_i, N=1213
- cap=50.0, 实际截断了5个标签(原51.74~56.76 -> 统一50.00)
- pos_weight范围: 4.10 ~ 50.00, 均值30.84
- sanity check: batch_size=4, num_labels=50, loss=10.97, 梯度无NaN
- 数值稳定性验证: BCEWithLogitsLoss在z=±100极端情况下依然稳定,
  手动Sigmoid+BCELoss会在z=100时产生NaN(1×log(0)问题)
