# 第一天：Python & 数据科学库速成攻坚

> 日期：20250825


## 学习计划

## 学习过程
主要用来记录学习过程中遇到的问题

### numpy

#### 安装numpy过程遇到报错问题
使用 `pip3 install numpy` 的过程中收到报错如下：

```bash
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try brew install
    xyz, where xyz is the package you are trying to
    install.
    
    If you wish to install a Python library that isn't in Homebrew,
    use a virtual environment:
    
    python3 -m venv path/to/venv
    source path/to/venv/bin/activate
    python3 -m pip install xyz
    
    If you wish to install a Python application that isn't in Homebrew,
    it may be easiest to use 'pipx install xyz', which will manage a
    virtual environment for you. You can install pipx with
    
    brew install pipx
```

##### 解决方案
使用虚拟环境安装

```bash
# 创建虚拟环境
python3 -m venv myenv

# 激活虚拟环境
source myenv/bin/activate

# 现在可以安全安装包
pip install numpy
```


### numpy数组中涉及的知识点

1. 数组形状
2. 索引与切片
3. 向量化运算
4. 矩阵运算

#### 矩阵乘法的计算方式不理解
数学中，矩阵乘法是一种根据两个矩阵得到第三个矩阵的二元运算，第三个矩阵即前两者的乘积，称为矩阵积。设A是n×m的矩阵，B是m×p的矩阵，则它们的矩阵积AB是n×p的矩阵。A中每一行的m个元素都与B中对应列的m个元素对应相乘，这些乘积的和就是AB中的一个元素。

> 上面这一系列的定义，像是公式，但是不太理解为什么是【A的第i行】和【B的第j列】相乘？。deepseek举了一下例子，让我能够更好的理解

```
请把自己想象成一个工厂的老板，你的工厂生产汽车。

矩阵A： 是你的 “订单表”。

每一行代表一个客户的订单。（客户1, 客户2...）

每一列代表生产一辆车需要的部件数量。（车轮、方向盘、引擎...）

A = [[2, 1, 3], # 客户1的订单：要2个车轮、1个方向盘、3个引擎 [1, 1, 2]] # 客户2的订单：要1个车轮、1个方向盘、2个引擎

矩阵B： 是你的 “成本表”。

每一行代表一个部件的成本。（车轮的成本、方向盘的成本...）

每一列代表一种货币。（美元、欧元...）

B = [[100, 90], # 一个车轮花费$100 / €90 [50, 40], # 一个方向盘花费$50 / €40 [200, 180]] # 一个引擎花费$200 / €180

现在，你这个老板想知道：“每个客户的总订单，用美元算是多少钱？用欧元算是多少钱？”

这个结果，就是你要的输出矩阵O。

接下来可以帮助理解为什么是【A的第i行】和【B的第j列】？

你想计算：客户i的订单，用货币j来计算的总成本是多少？

【A的第i行】是什么？

这是客户i的完整订单清单。比如i=0（客户1），他的订单就是[2, 1, 3]（2个轮子，1个方向盘，3个引擎）。

【B的第j列】是什么？

这是每个部件用货币j计算的单价。比如j=0（美元），所有部件的美元单价就是[100, 50, 200]（轮子$100，方向盘$50，引擎$200）。

如何计算客户1（i=0）的订单总价（美元j=0）？

逻辑：把他订单里的每个部件数量，乘上对应的美元单价，然后全部加起来！
计算：(2个轮子 * $100/轮) + (1个方向盘 * $50/方向盘) + (3个引擎 * $200/引擎) = $200 + $50 + $600 = $850

你看，这个计算过程，不就是【A的第0行】[2, 1, 3] 和 【B的第0列】[100, 50, 200] 的点积吗？

(2*100) + (1*50) + (3*200) = 850

所以，输出矩阵O的第[i, j]个元素（O[i, j]），自然就是【A的第i行】和【B的第j列】的点积结果。

O[0, 0] = 850 （客户1的订单总价，美元）

同样，O[0, 1] 就是客户1的订单总价，用欧元算。拿A的第0行 [2, 1, 3] 和 B的第1列 [90, 40, 180] 点积。

O[1, 0] 就是客户2的订单总价，用美元算。拿A的第1行 [1, 1, 2] 和 B的第0列 [100, 50, 200] 点积。
```

### Pandas
#### 不知道如何使用iris.data的数据
因为下载的数据里面没有数据名称，都是一堆csv格式的数据，虽然我学会了使用df.head()等方法，但是打印出来的数据根本看不懂

运行出来的数据是这样的：
```
	5.1	3.5	1.4	0.2	Iris-setosa
0	4.9	3.0	1.4	0.2	Iris-setosa
1	4.7	3.2	1.3	0.2	Iris-setosa
2	4.6	3.1	1.5	0.2	Iris-setosa
3	5.0	3.6	1.4	0.2	Iris-setosa
4	5.4	3.9	1.7	0.4	Iris-setosa
```

##### 解决方案
其实挺简单的，就是自己定义好每一列的名称，将它传递给`read_csv`的方法
```python
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

df = pd.read_csv('../assets/datas/iris/iris.data', names=column_names)

# 查看前5行数据，确认加载成功
print("数据前5行：")
df.head()
```

运行以上代码就可以看到如下数据：
```
   sepal_length  sepal_width  petal_length  petal_width      species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
```

## 学习成果