# Decision Tree

- init: 生成一颗DT: x, y, method, pruning, 
- 确定使用的择优方法method: infomation_gain, gain_ratio, gain-ratio_heuristic, gini_index
- 计算择优值
- 剪枝pruning: None, pre, post
- 连续值分割
- 空数据处理
- def 预测方法
- 子树连接
- string数据处理
- 属性: method, pruning, sub_trees, 

## step one:

- information_gain方法
  - 离散
- 离散数据处理: 映射为0~n
- 子树连接
- 预测函数
## step two

- information_gain
  - 连续
- gain_ratio
- gain_ratio_heuristic
- gini_index

## step three

- 空数据处理
    - 拟合带有空值的数据
    - 预测带有空值的数据
    - 预测没有出现过的数据
- string数据映射

## 剪枝
- prepruning
- postpruning