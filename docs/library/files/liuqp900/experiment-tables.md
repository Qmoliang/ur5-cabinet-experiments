# 冻结900次实验：逐例重算表格

本页仅从已发布索引重算描述统计，不改变原始判定。源提交：`0f402010bb8335df661fdace8745106b7ee5aede`。

[论文初稿](paper-draft.md) · [机器可读统计](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/paper-evidence.json)

## 总体结果

| 组 | 原始到达 | 物理安全到达 | 严格成功 |
| --- | --- | --- | --- |
| C | 175/300 | 174/300（58.00%） | 128/300（42.67%） |
| A | 186/300 | 186/300（62.00%） | 186/300（62.00%） |
| B | 294/300 | 294/300（98.00%） | 294/300（98.00%） |

## 逐例新增与退化

| 比较 | 口径 | 共同成功 | 新增 | 退化 | 共同失败 | 净增 |
| --- | --- | --- | --- | --- | --- | --- |
| C→A | 严格 | 127 | 59 | 1 | 113 | 58 |
| C→A | 物理安全到达 | 171 | 15 | 3 | 111 | 12 |
| A→B | 严格 | 185 | 109 | 1 | 5 | 108 |
| A→B | 物理安全到达 | 185 | 109 | 1 | 5 | 108 |
| C→B | 严格 | 128 | 166 | 0 | 6 | 166 |
| C→B | 物理安全到达 | 173 | 121 | 1 | 5 | 120 |

## 15项任务

| 任务（每组20次） | C 物理/严格 | A 物理/严格 | B 物理/严格 |
| --- | --- | --- | --- |
| drawer | 14/8 | 18/18 | 18/18 |
| cage | 7/0 | 15/15 | 20/20 |
| cage_wide | 20/20 | 20/20 | 20/20 |
| exit | 10/0 | 11/11 | 20/20 |
| iris_4shelves_t0 | 0/0 | 0/0 | 20/20 |
| iris_4shelves_t2 | 0/0 | 0/0 | 20/20 |
| iris_bins_t0 | 18/0 | 19/19 | 19/19 |
| iris_bins_t1 | 0/0 | 0/0 | 18/18 |
| iris_bins_t2 | 20/20 | 20/20 | 20/20 |
| iris_bins_t3 | 20/20 | 19/19 | 20/20 |
| iris_ur3_shelf_t0 | 20/20 | 20/20 | 20/20 |
| iris_ur3_shelf_t1 | 20/20 | 20/20 | 20/20 |
| iris_ur3_shelf_t2 | 20/20 | 20/20 | 20/20 |
| wall2_topm10 | 4/0 | 3/3 | 20/20 |
| wall2_topm15 | 1/0 | 1/1 | 19/19 |

## 完整耗时

| 组 | 完整单次中位数/s | P90/s | 最大值/s | 含共享准备计费中位数/s | 终态初始化中位数/s | 事后审计中位数/s |
| --- | --- | --- | --- | --- | --- | --- |
| C | 15.192 | 1057.509 | 1200.305 | 17.753 | 0.000 | 25.679 |
| A | 41.134 | 184.455 | 272.429 | 43.944 | 0.000 | 26.394 |
| B | 144.661 | 352.447 | 602.915 | 146.202 | 79.490 | 12.139 |

## 全部六个B组失败

| B组失败案例 | 末误差/mm | 仿真/s | 完整墙钟/s | 扩展激活 | 绕行次数 | 重复检测次数 |
| --- | --- | --- | --- | --- | --- | --- |
| [drawer s08](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/#case=online__drawer__s08&arm=B) | 323.152788 | 300 | 453.513 | 是 | 2 | 0 |
| [drawer s14](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/#case=online__drawer__s14&arm=B) | 417.761618 | 300 | 464.987 | 是 | 2 | 0 |
| [iris_bins_t0 s17](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/#case=online__iris_bins_t0__s17&arm=B) | 1.380197 | 300 | 160.882 | 否 | 0 | 0 |
| [iris_bins_t1 s13](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/#case=online__iris_bins_t1__s13&arm=B) | 339.562401 | 300 | 312.484 | 是 | 1 | 0 |
| [iris_bins_t1 s15](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/#case=online__iris_bins_t1__s15&arm=B) | 304.413178 | 300 | 391.550 | 是 | 0 | 0 |
| [wall2_topm15 s11](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/#case=online__wall2_topm15__s11&arm=B) | 327.438423 | 300 | 218.003 | 是 | 1 | 0 |
