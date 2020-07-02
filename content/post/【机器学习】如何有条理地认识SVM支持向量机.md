---
title: "【机器学习】有条理地认识SVM支持向量机"
date: 2020-06-13T10:31:59+08:00
tags: ["数据挖掘"]
---﻿
SVM 具有完善的数学理论，虽然如今工业界用到的不多，但还是很值得初学者学习的. 为了方便以后回顾之用, 将本人学习过程中的理解以及其他大神的成果整理成本文, 本文主要面向两类读者, ==A类读者==: 希望了解简单的算法流程即可, 不打算深入了解理论推导过程的, 或已学习过此算法, 但记忆模糊希望简单回顾的; ==B类读者==:希望深入学习理论推导过程的. A类读者从下一段开始看到''目录''前为止即可, B类读者则从''目录''开始看到最后, 请各位对号入座, 这样做目的是为了提高不同需求的读者的学习效率, 节省你们的时间, 其中A类是本人总结概括的, B类借鉴了知乎一位大神的文章, 在一些难懂的地方扩展了一下, 希望各位能更加容易读懂 !

==**A类**==:
SVM正常情况就是用于分类问题的一个算法, 那么分类问题从简单到复杂可以划分为以下3种(以二分类为例):
1.	硬间隔(理想情况): 两类点分隔明显
线性可分: 一条线能把两类点分开, 一类是$wx_i+b<0$ ,一类是$wx_j+b>0$ 
最大间隔超平面: 扩展到多维中, 分割线离两类点最近的点的距离最大化, 即这个平面尽可能在中间, 离两类点尽可能远
支持向量: 距离超平面最近的点
问题转化为求这个w和b, 通过拉格朗日乘数法, 强对偶性转化为二次规划问题, 再通过SMO算法求解, SMO算法思想是每次只优化一个参数，其他参数先固定住，仅求当前这个优化参数的极值, 最后w和b求出后即可得到模型f(x):
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629170711521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
 
2.	软间隔: 两类点有少量渗透到对方中, 允许个别点在间隔带中, 此时引入松弛变量来衡量这个"允许"的程度, 以减少过拟合, 同样通过拉格朗日和SMO来求解即可
3.	核函数: 样本点不是线性可分的, 映射到高维空间线性可分, 因此称为非线性SVM
由于上述提到的计算过程中部分步骤有内积计算等复杂度较高的过程, 因此扩维后再计算的量特别大,那么核函数作用就是可以先计算好再扩维, 降低计算量
常用核函数有线性核函数, 多项式核函数, 高斯核函数, 其中高斯核函数需要调参

**优点:** 小样本下,可以解决高维问题, 即大型特征空间, 一般选择线性核函数，效果不好上高斯核函数，高斯核函数可以映射到无限维空间
**缺点:** 对非线性问题没有通用解决方案，有时候很难找到一个合适的核函数, 当观测样本很多时，效率并不是很高, 且SVM对缺失数据敏感
**多分类问题解决思路:** 
1)one-other,即有几个分类就建几个SVM, 样本都过一遍SVM,找出成功分类且离超平面最远的那一类
2)one-one,即两两分类之间分别建立SVM, 两两间比较样本点, 取出分类次数最多那一类
**总结**: 看到这里的A类读者大概对SVM有一点概念就可以了, 总的来说SVM就是想方设法地作出一些线或超平面来划分我们的样本点达到分类的目的, 而且是用很系统完善的数学角度去做, 非常谨慎, 有理有据,理论上任何分类的问题都能解决, 但致命点在于计算量大, 那么接下来A类读者就可以去找一些SVM的实例代码, 照着实践一下, 知道各个参数的含义, 会用就可以了.

==**B类**==:
@[TOC](目录)
# 1. SVM算法思想
## 1.1 线性可分
首先我们先来了解下什么是线性可分。
在二维空间上，两类点被一条直线完全分开叫做线性可分,直观地理解就是如下图所示(当然这条线不唯一):
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629010622235.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
严格的数学定义是:
$D_0$和$D_1$是 n 维欧氏空间中的两个点集。如果存在 n 维向量 w 和实数 b，使得所有属于$D_0$的点$x_i$都有$wx_i+b<0$ ，而对于所有属于$D_1$ 的点$x_j$ 则有$wx_j+b>0$，则我们称$D_0$和$D_1$线性可分。
## 1.2 最大间隔超平面
上面那个图是二维, 比较好理解, 那么从二维扩展到多维空间中呢，将$D_0$和$D_1$完全正确地划分开的$wx+b=0$  就成了一个超平面。
无限个这样的平面中, 如果要找最佳超平面，以最大间隔把两类样本分开的超平面，那么这个平面称之为最大间隔超平面。
最大间隔超平面有以下特点:
**a**.两类样本分别分割在该超平面的两侧；
**b**.两侧距离超平面最近的样本点到超平面的距离被==最大化==了。
## 1.3 支持向量
样本中距离超平面最近的一些点，这些点叫做==支持向量==(支撑向量), 如下图:![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629012217481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
## 1.4 最优化问题
SVM 想要的就是找到最大间隔超平面。任意超平面可以用下面这个线性方程来描述：
 $w^Tx+b=0$
 二维空间点 $(x,y)$到直线$Ax+By+C=0$的距离公式是：

<font size=5>$\frac{|Ax+By+C|}{\sqrt{A^2+B^2}}$</font>

扩展到 n 维空间后，点$x=(x_1,x_2,...,x_n)$到直线$w^Tx+b=0$的距离为：

<font size=5>$\frac{|w^Tx+b|}{||w||}$</font>

其中$||w||=\sqrt{w_1^2+w_2^2+...+w_n^2}$ 。

如下图所示，显然我们明白根据支持向量的定义，假设支持向量到超平面的距离为 d，那么其他点到超平面的距离就大于 d。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629014509300.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
设其中一类点的标签值为$y=1$,另一类为$y=-1$, 就有以下公式:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629015122574.png#pic_center)
可以转化为:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629015149482.png#pic_center)
其中$||w||d$是正数,且它的大小对结果无影响, 那么为了推导方便, 令它等于1, 有:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629201643813.png#pic_center)
把两个式子合并起来就是:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629202057263.png#pic_center)
此时令等号成立, y分别取-1或1时, 就得到最大间隔超平面两边的==两个超平面==, 如图:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629202336256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
还记得刚才提到的n维空间中, 点到直线的距离公式吗?那么这里支持向量到最大间隔超平面的距离就是:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629202719945.png#pic_center)
上面提到$y(w^Tx+b)≥1$, 即$y(w^Tx+b)>0$, 因此$|w^Tx+b|=y(w^Tx+b)$, 所以刚才的距离公式等于:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629203817239.png#pic_center)
给它乘上一个系数2, 对目标函数是没有影响的, 仅仅为了方便推导, 然后我们做这么多不就是为了让这个d最大吗, 因此得到:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629204324599.png#pic_center)
这个是求最大值问题, 而$y(w^Tx+b)=1$,因此有:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629204420755.png#pic_center)
求一个式子的最大等于求它的最小值:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629204450786.png#pic_center)
还记得$||w||$是二范数吗, 是有根号的, 那么为了去掉根号, 就有:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629204520722.png#pic_center)
到这里先捋一下思路, 我们目的是==想找到最大间隔超平面==, 转化为→==求支持向量到最大间隔超平面的距离d的最大值==→==一步一步化简函数==→==求上面这个式子的最小值==:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629205047120.png#pic_center)
显然这是一个不等式约束的优化问题, 高等数学的内容, 还记得解决思路吗, 就是把不等式约束转化为等式约束再求解, 下面我们进一步看看怎么做.
# 2. 理论推导求解
## 2.1 拉格朗日乘子法
先回顾一下拉格朗日乘子法对等式约束的优化问题是怎么求解的:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629214156348.png#pic_center)
令$L(x,λ)=f(x)+\sum\limits_{k=1}^{l}\lambda_kh_k(x)$, 函数$L(x,\lambda)$称为Lagrange函数,$\lambda$称为Lagrange乘子, 没有非负要求.
利用必要条件找到可能的极值点:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629214744762.png#pic_center)
具体是否为极值点需根据问题本身的具体情况检验。这个方程组称为等式约束的极值必要条件。

等式约束下的 Lagrange 乘数法引入了$l$个 Lagrange 乘子，我们将 $x_i$与 $\lambda_k$ 一视同仁，把 $\lambda_k$也看作优化变量，共有$(n+l)$个优化变量。
而我们现在面对的是不等式优化问题，针对这种情况其主要思想是将不等式约束条件转变为等式约束条件，引入**松弛变量**，将松弛变量也视为优化变量, 整个过程如下图所示:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629215119131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
以我们的例子为例：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629221315797.png#pic_center)
我们引入松弛变量$a_i^2$得到 $h_i(w,a_i)=g_i(w)+a_i^2=0$ 。这里加平方主要为了不再引入新的约束条件，如果只引入 $a_i$那我们必须要保证$a_i≥0$才能保证 $h_i(w,a_i)=0$，这不符合我们的意愿。

由此我们将不等式约束转化为了等式约束，并得到 Lagrange 函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629221701794.png#pic_center)

由等式约束优化问题极值的必要条件对其求解，联立方程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629221755856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
（为什么取$\lambda_i≥0$ ，可以通过几何性质来解释，有兴趣的同学可以查下 KKT 的证明）。
针对 $\lambda_ia_i=0$我们有两种情况：

情形一： $\lambda_i=0,a_i≠0$

由于 $\lambda_i=0$，因此约束条件$g_i(w)$不起作用，且 $g_i(w)<0$

情形二： $\lambda_i≠0,a_i=0$

此时$g_i(w)=0$且$\lambda_i>0$，可以理解为约束条件$g_i(w)$起作用了，且$g_i(w)=0$

综合可得： $\lambda_ig_i(w)=0$，且在约束条件起作用时 $\lambda_i>0,g_i(w)=0$；约束不起作用时$\lambda_i=0,g_i(w)<0$

由此方程组转换为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629222643107.png#pic_center)

以上便是不等式约束优化优化问题的 KKT(Karush-Kuhn-Tucker) 条件，$\lambda_i$称为 KKT 乘子。

这个式子告诉了我们什么事情呢？

直观来讲就是，支持向量$g_i(w)=0$，所以$\lambda_i>0$即可。而其他向量$g_i(w)<0,\lambda_i=0$。

我们原本问题是要求：$min\frac{1}{2}||w||^2$，即求 $minL(w,\lambda,a)$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629223341788.png#pic_center)

由于$\sum\limits_{i=1}^{n}\lambda_ia_i^2≥0$，故我们将问题转换为：$minL(w, \lambda)$：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629223628574.png#pic_center)

假设找到了最佳参数是的目标函数取得了最小值 p。即$\frac{1}{2}||w||^2=p$。而根据$\lambda_i≥0$，可知$\sum\limits_{i=1}^{n}\lambda_ig_i(w)≤0$，因此 $L(w, \lambda)≤p$，为了找到最优的参数$\lambda$ ，使得$L(w,\lambda)$接近 p，故问题转换为:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020062922431585.png#pic_center)

故我们的最优化问题转换为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629224344506.png#pic_center)
除了上面的理解方式，我们还可以有另一种理解方式： 由于 $\lambda_i≥0$ ，有:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200629224833651.png#pic_center)

所以$min(∞,\frac{1}{2}||w||^2)=\frac{1}{2}||w||^2$，即转化后的式子和原来的式子也是一样的。
## 2.2 强对偶性
对偶问题其实就是将：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630033908469.png#pic_center)
变成了：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020063003395295.png#pic_center)
假设有个函数 [公式] 我们有：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630034042197.png#pic_center)

也就是说，最大的里面挑出来的最小的也要比最小的里面挑出来的最大的要大。这关系实际上就是弱对偶关系，而强对偶关系是当等号成立时，即：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630034108509.png#pic_center)
如果$f$ 是凸优化问题，强对偶性成立。而我们之前求的 KKT 条件是强对偶性的**充要条件**。
# 3. SVM优化
我们已知 SVM 优化的主问题是：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630034818700.png#pic_center)

那么求解线性可分的 SVM 的步骤为：

**步骤 1：**

构造拉格朗日函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630034846836.png#pic_center)

**步骤 2：**

利用强对偶性转化：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630034905598.png#pic_center)
现对参数 w 和 b 求偏导数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630034922613.png#pic_center)

得到：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630034959348.png#pic_center)

我们将这个结果代入到函数中可得：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630035019129.png#pic_center)

也就是说：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630035036459.png#pic_center)

**步骤 3：**

由步骤 2 得：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630035204284.png#pic_center)

我们可以看出来这是一个二次规划问题，问题规模正比于训练样本数，我们常用 SMO(Sequential Minimal Optimization) 算法求解。

SMO(Sequential Minimal Optimization)，序列最小优化算法，其核心思想非常简单：每次只优化一个参数，其他参数先固定住，仅求当前这个优化参数的极值。我们来看一下 SMO 算法在 SVM 中的应用。

我们刚说了 SMO 算法每次只优化一个参数，但我们的优化目标有约束条件：$\sum\limits_{i=1}^{n}\lambda_iy_i=0$，没法一次只变动一个参数。所以我们选择了一次选择两个参数。具体步骤为：

选择两个需要更新的参数$\lambda_i$和 $\lambda_j$ ，固定其他参数。于是我们有以下约束：
这样约束就变成了：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630035432910.png#pic_center)

其中$c=-\sum\limits_{k≠i,j}^{}\lambda_ky_k$ ，由此可以得出$\lambda_j=\frac{c-\lambda_iy_i}{y_i}$ ，也就是说我们可以用 $\lambda_i$ 的表达式代替$\lambda_j$ 。这样就相当于把目标问题转化成了仅有一个约束条件的最优化问题，仅有的约束是$\lambda_i≥0$。

2. 对于仅有一个约束条件的最优化问题，我们完全可以在 $\lambda_i$上对优化目标求偏导，令导数为零，从而求出变量值 ${\lambda_i}_{new}$ ，然后根据${\lambda_i}_{new}$ 求出${\lambda_j}_{new}$。

3. 多次迭代直至收敛。

通过 SMO 求得最优解 $\lambda^*$ 。

**步骤 4 ：**

我们求偏导数时得到：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630035926331.png#pic_center)

由上式可求得 w。

我们知道所有$\lambda_i>0$对应的点都是支持向量，我们可以随便找个支持向量，然后带入：$y_s(wx_s+b)=1$，求出 b 即可，

两边同乘$y_s$，得 $y_s^2(wx_s+b)=y_s$

因为$y_s^2=1$，所以：$b=y_s-wx_s$

为了更具鲁棒性，我们可以求得支持向量的均值：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630040145362.png#pic_center)

**步骤 5：** w 和 b 都求出来了，我们就能构造出最大分割超平面： $w^Tx+b=0$

分类决策函数： $f(x)=sign(w^Tx+b)$

其中 $sign(·)$为阶跃函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630040320486.png#pic_center)

将新样本点导入到决策函数中既可得到样本的分类。
# 4. 实际情况求解
## 4.1 解决问题
在实际应用中，更一般地, 完全线性可分的样本是很少的，如果遇到了不能够完全线性可分的样本，我们应该怎么办？比如下面这个：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630085524679.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
于是我们就有了软间隔，相比于硬间隔的苛刻条件，软间隔允许个别样本点出现在间隔带里面，比如：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630085541138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
我们允许部分样本点不满足约束条件：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630085640979.png#pic_center)
为了度量这个间隔软到何种程度，我们为每个样本引入一个松弛变量 $\xi_i$，令$\xi_i≥0$ ，且$1-y_i(w^Tx_i+b)-\xi_i≤0$。对应如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630090026427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)
## 4.2 优化目标及求解
增加软间隔后我们的优化目标变成了：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630090236506.png#pic_center)

其中 C 是一个大于 0 的常数，可以理解为错误样本的惩罚程度，若 C 为无穷大，$\xi_i$ 必然无穷小，如此一来线性 SVM 就又变成了线性可分 SVM；当 C 为有限值的时候，才会允许部分样本不遵循约束条件。

接下来我们将针对新的优化目标求解最优化问题：

**步骤 1：**

构造拉格朗日函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630090352795.png#pic_center)

其中$\lambda_i$ 和 $μ_i$是拉格朗日乘子，w、b 和$\xi_i$是主问题参数。

根据强对偶性，将对偶问题转换为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630090737272.png#pic_center)

**步骤 2：**

分别对主问题参数w、b 和 $\xi_i$ 求偏导数，并令偏导数为 0，得出如下关系：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630091138728.png#pic_center)

将这些关系带入拉格朗日函数中，得到：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630091200641.png#pic_center)

最小化结果只有 $\lambda$而没有 $μ$，所以现在只需要最大化 $\lambda$就好：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630091306533.png#pic_center)

我们可以看到这个和硬间隔的一样，只是多了个约束条件。

然后我们利用 SMO 算法求解得到拉格朗日乘子$\lambda^*$。

**步骤 3 ：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630091351850.png#pic_center)

然后我们通过上面两个式子求出 w 和 b，最终求得超平面$w^Tx+b=0$，

**这边要注意一个问题，在间隔内的那部分样本点是不是支持向量？**

我们可以由求参数 w 的那个式子可看出，只要$\lambda_i>0$的点都能够影响我们的超平面，因此都是支持向量。
# 5. 核函数
## 5.1 线性不可分
刚才的硬间隔和软间隔还算比较好解决的, 样本都是完全线性可分或者大部分样本点的线性可分, 但我们实际可能会碰到的更复杂的一种情况是样本点不是线性可分的，比如：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630092038880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)

这种情况的解决方法就是：将二维线性不可分样本映射到高维空间中，让样本点在高维空间线性可分，比如：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020063009211850.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E3Mzg4Nzg3,size_16,color_FFFFFF,t_70#pic_center)

对于在有限维度向量空间中线性不可分的样本，我们将其映射到更高维度的向量空间里，再通过间隔最大化的方式，学习得到支持向量机，就是非线性 SVM。

我们用 x 表示原来的样本点，用 $\varphi(x)$表示 x 映射到特征新的特征空间后到新向量。那么分割超平面可以表示为：$f(x)=w\varphi(x)+b$。

对于非线性 SVM 的对偶问题就变成了：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630092704733.png#pic_center)

可以看到与线性 SVM 唯一的不同就是：之前的 $(x_i·x_j)$变成了 $(\varphi(x_i)·\varphi(x_j))$。
## 5.2 核函数的作用
我们不禁有个疑问：只是做个内积运算，为什么要有核函数的呢？

这是因为低维空间映射到高维空间后维度可能会很大，如果将全部样本的点乘全部计算好，这样的计算量太大了。

但如果我们有这样的一核函数$k(x,y)=(\varphi(x)·\varphi(y))$ ，$x_i$ 与$x_j$ 在特征空间的内积等于它们在原始样本空间中通过函数 $k(x,y)$ 计算的结果，我们就不需要计算高维甚至无穷维空间的内积了。

举个例子：假设我们有一个多项式核函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630095021777.png#pic_center)

带进样本点的后：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630095040191.png#pic_center)

而它的展开项是：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630095057616.png#pic_center)

如果没有核函数，我们则需要把向量映射成：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630095117281.png#pic_center)

然后在进行内积计算，才能与多项式核函数达到相同的效果。

可见核函数的引入一方面减少了我们计算量，另一方面也减少了我们存储数据的内存使用量。
## 5.3 常见核函数
我们常用核函数有：

**线性核函数**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630095411273.png#pic_center)

**多项式核函数**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630095424697.png#pic_center)

**高斯核函数**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200630095437850.png#pic_center)

这三个常用的核函数中只有高斯核函数是需要调参的。
# 6. SVM优缺点
## 6.1 优点
1. 有严格的数学理论支持，可解释性强，不依靠统计方法，从而简化了通常的分类和回归问题；
2. 能找出对任务至关重要的关键样本（即：支持向量）；
3. 采用核技巧之后，可以处理非线性分类/回归任务；
4. 最终决策函数只由少数的支持向量所确定，计算的复杂性取决于支持向量的数目，而不是样本空间的维数，这在某种意义上避免了“维数灾难”。
## 6.2 缺点
1. 训练时间长。当采用 SMO 算法时，由于每次都需要挑选一对参数，因此时间复杂度为 $O(N^2)$，其中 N 为训练样本的数量；
2. 当采用核技巧时，如果需要存储核矩阵，则空间复杂度为 $O(N^2)$；
3. 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高。

因此支持向量机目前只适合小批量样本的任务，无法适应百万甚至上亿样本的任务。

**参考:**
[详解SVM](https://zhuanlan.zhihu.com/p/77750026?utm_source=wechat_session)
