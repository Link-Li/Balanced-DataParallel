# Balanced-DataParallel
这里是改进了pytorch的DataParallel, 用来平衡第一个GPU的显存使用量

本代码来自transformer-XL:https://github.com/kimiyoung/transformer-xl

代码不是本人写的, 但是感觉很好用, 就分享一下.

# 怎么使用:

&emsp;&emsp;这个 `BalancedDataParallel` 类使用起来和 `DataParallel` 类似, 下面是一个示例代码:

```
my_net = MyNet()
my_net = BalancedDataParallel(gpu0_bsz // acc_grad, my_net, dim=0).cuda()
```

&emsp;&emsp;这里包含三个参数, 第一个参数是第一个GPU要分配多大的batch_size, 但是要注意, 如果你使用了梯度累积, 那么这里传入的是每次进行运算的实际batch_size大小. 举个例子, 比如你在3个GPU上面跑代码, 但是一个GPU最大只能跑3条数据, 但是因为0号GPU还要做一些数据的整合操作, 于是0号GPU只能跑2条数据, 这样一算, 你可以跑的大小是2+3+3=8, 于是你可以设置下面的这样的参数:

```
batch_szie = 8
gpu0_bsz = 2
acc_grad = 1
my_net = MyNet()
my_net = BalancedDataParallel(gpu0_bsz // acc_grad, my_net, dim=0).cuda()
```

&emsp;&emsp;这个时候突然想跑个batch size是16的怎么办呢, 那就是4+6+6=16了, 这样设置累积梯度为2就行了:


```
batch_szie = 16
gpu0_bsz = 4
acc_grad = 2
my_net = MyNet()
my_net = BalancedDataParallel(gpu0_bsz // acc_grad, my_net, dim=0).cuda()

```

### 各个版本的data_parallel

- data_parallel.py: 原作者的代码, 但是使用的时候发现, 如果batch size设置的小于GPU的数量, 会导致最后一个批次的数据分配的不足以所有的GPU分配, 然后报错.

- data_parallel_my.py: 我稍微改了一点, 然后稍微测试了一下, 应该是解决了上面的问题.

- data_parallel_my_v2.py：上面第一个版本的修改，导致无法设置gpu0_bsz=0，这个版本应该是修复这个问题了
