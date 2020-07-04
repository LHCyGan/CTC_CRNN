## CRNN基本网络结构

![img](https://pic3.zhimg.com/v2-7ed5c65fe79dce49f006a9171cc1a80e_r.jpg)图4 CRNN网络结构（此图按照本文给出的github实现代码画的）

整个CRNN网络可以分为三个部分：

假设输入图像大小为 ![[公式]](https://www.zhihu.com/equation?tex=%2832%2C+100%2C3%29)，注意提及图像都是 ![[公式]](https://www.zhihu.com/equation?tex=%28%5Ctext%7BHeight%7D%2C%5Ctext%7BWidth%7D%2C%5Ctext%7BChannel%7D%29) 形式。

- Convlutional Layers

这里的卷积层就是一个普通的CNN网络，用于提取输入图像的Convolutional feature maps，即将大小为 ![[公式]](https://www.zhihu.com/equation?tex=%2832%2C+100%2C3%29) 的图像转换为 ![[公式]](https://www.zhihu.com/equation?tex=%281%2C25%2C512%29) 大小的卷积特征矩阵，网络细节请参考本文给出的实现代码。

- Recurrent Layers

这里的循环网络层是一个深层双向LSTM网络，在卷积特征的基础上继续提取文字序列特征。对RNN不了解的读者，建议参考：

[完全解析RNN, Seq2Seq, Attention注意力机制zhuanlan.zhihu.com![图标](https://pic4.zhimg.com/v2-2d34ffd69bc1d4d1c9231d0562db2fcf_180x120.jpg)](https://zhuanlan.zhihu.com/p/51383402)

所谓深层RNN网络，是指超过两层的RNN网络。对于单层双向RNN网络，结构如下：

![img](https://pic4.zhimg.com/v2-9f5125e0c99924d2febf25bafd019d6f_r.jpg)图5 单层双向RNN网络

而对于深层双向RNN网络，主要有2种不同的实现：

```python
tf.nn.bidirectional_dynamic_rnn
```

![img](https://pic3.zhimg.com/v2-c0132f0b748eb031c696dae3019a2d82_r.jpg)图6 深层双向RNN网络

```python
tf.contrib.rnn.stack_bidirectional_dynamic_rnn
```

![img](https://pic2.zhimg.com/v2-00861a152263cff8b94525d8b8945ee9_r.jpg)图7 stack形深层双向RNN网络

在CRNN中显然使用了第二种stack形深层双向结构。

由于CNN输出的Feature map是![[公式]](https://www.zhihu.com/equation?tex=%281%2C25%2C512%29)大小，所以对于RNN最大时间长度 ![[公式]](https://www.zhihu.com/equation?tex=T%3D25) （即有25个时间输入，每个输入 ![[公式]](https://www.zhihu.com/equation?tex=x_t) 列向量有 ![[公式]](https://www.zhihu.com/equation?tex=D%3D512) ）。

- Transcription Layers

将RNN输出做softmax后，为字符输出。

**关于代码中输入图片大小的解释：**

在本文给出的实现中，为了将特征输入到Recurrent Layers，做如下处理：

- 首先会将图像在固定长宽比的情况下缩放到 ![[公式]](https://www.zhihu.com/equation?tex=32%5Ctimes+W%5Ctimes3) 大小（ ![[公式]](https://www.zhihu.com/equation?tex=W) 代表任意宽度）
- 然后经过CNN后变为 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+%28W%2F4%29%5Ctimes512)
- 针对LSTM设置 ![[公式]](https://www.zhihu.com/equation?tex=T%3D%28W%2F4%29) ，即可将特征输入LSTM。

所以在处理输入图像的时候，建议在保持长宽比的情况下将高缩放到 ![[公式]](https://www.zhihu.com/equation?tex=32)，这样能够尽量不破坏图像中的文本细节（当然也可以将输入图像缩放到固定宽度，但是这样由于破坏文本的形状，肯定会造成性能下降）。

## 考虑训练Recurrent Layers时的一个问题：

![img](https://pic2.zhimg.com/v2-5803de0cd9eb4e20f6a722e02b196809_r.jpg)图8 感受野与RNN标签的关系

对于Recurrent Layers，如果使用常见的Softmax cross-entropy loss，则每一列输出都需要对应一个字符元素。那么训练时候每张样本图片都需要标记出每个字符在图片中的位置，再通过CNN感受野对齐到Feature map的每一列获取该列输出对应的Label才能进行训练，如图9。

在实际情况中，标记这种对齐样本非常困难（除了标记字符，还要标记每个字符的位置），工作量非常大。另外，由于每张样本的字符数量不同，字体样式不同，字体大小不同，导致每列输出并不一定能与每个字符一一对应。

当然这种问题同样存在于语音识别领域。例如有人说话快，有人说话慢，那么如何进行语音帧对齐，是一直以来困扰语音识别的巨大难题。

![img](https://pic3.zhimg.com/80/v2-11fcc223d3288932fd15a1d2a26f2c26_720w.jpg)图9

所以CTC提出一种对不需要对齐的Loss计算方法，用于训练网络，被广泛应用于文本行识别和语音识别中。

## Connectionist Temporal Classification(CTC)详解

在分析过程中尽量保持和原文符号一致。

[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networksftp.idsia.ch](https://link.zhihu.com/?target=ftp%3A//ftp.idsia.ch/pub/juergen/icml2006.pdf)

整个CRNN的流程如图10。先通过CNN提取文本图片的Feature map，然后将每一个channel作为 ![[公式]](https://www.zhihu.com/equation?tex=D%3D512) 的时间序列输入到LSTM中。

![img](https://pic2.zhimg.com/v2-6e2120edda0684a2a654d0627ad13591_r.jpg)图10 CRNN+CTC框架

为了说明问题，我们定义：

- CNN Feature map

Feature map的每一列作为一个时间片输入到LSTM中。设Feature map大小为 ![[公式]](https://www.zhihu.com/equation?tex=m%5Ccdot+T) （图11中 ![[公式]](https://www.zhihu.com/equation?tex=m%3D512) ，![[公式]](https://www.zhihu.com/equation?tex=T%3D25) ）。下文中的时间序列 ![[公式]](https://www.zhihu.com/equation?tex=t) 都从 ![[公式]](https://www.zhihu.com/equation?tex=t%3D1) 开始，即 ![[公式]](https://www.zhihu.com/equation?tex=1%5Cleq+t+%5Cleq+T) 。

定义为：

![[公式]](https://www.zhihu.com/equation?tex=x%3D%28x%5E1%2Cx%5E2%2C...%2Cx%5ET%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=x) 每一列 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et) 为：

![[公式]](https://www.zhihu.com/equation?tex=x%5Et%3D%28x_1%5Et%2Cx_2%5Et%2C...%2Cx_m%5Et%29%5C%5C)

- LSTM

LSTM的每一个时间片后接softmax，输出 ![[公式]](https://www.zhihu.com/equation?tex=y) 是一个后验概率矩阵，定义为：

![[公式]](https://www.zhihu.com/equation?tex=y%3D%28y%5E1%2Cy%5E2%2C...%2Cy%5Et%2C...%2Cy%5ET%29%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=y) 的每一列 ![[公式]](https://www.zhihu.com/equation?tex=y%5Et) 为：

![[公式]](https://www.zhihu.com/equation?tex=y%5Et%3D%28y%5Et_1%2Cy%5Et_2%2C...%2Cy%5Et_n%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 代表需要识别的字符集合长度。由于 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5Et) 是概率，所以服从概率假设：![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bk%7D%7By_k%5Et%7D%3D1)

对 ![[公式]](https://www.zhihu.com/equation?tex=y) 每一列进行 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bargmax%7D%28%29) 操作，即可获得每一列输出字符的类别。

那么LSTM可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=y%3D%5Ctexttt%7BNET%7D_w%28x%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=w) 代表LSTM的参数。LSTM在输入和输出间做了如下变换：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctexttt%7BNET%7D_w%3A%28R%5Em%29%5ET%5Crightarrow%28R%5En%29%5ET%5C%5C)

![img](https://pic1.zhimg.com/v2-7a20ee2f70e41a8662fe89fc8773ead0_r.jpg)图11

- 空白blank符号

如果要进行 ![[公式]](https://www.zhihu.com/equation?tex=L%3D%5C%7Ba%2Cb%2Cc%2C...%2Cx%2Cy%2Cz%5C%7D) 的26个英文字符识别，考虑到有的位置没有字符，定义插入blank的字符集合：

![[公式]](https://www.zhihu.com/equation?tex=L%27%3DL%5Ccup+%5C%7B%5Ctext%7Bblank%7D%5C%7D%5C%5C)

其中blank表示当前列对应的图像位置没有字符（下文以![[公式]](https://www.zhihu.com/equation?tex=-)符号表示blank）。

- 关于![[公式]](https://www.zhihu.com/equation?tex=B) 变换

定义变换 ![[公式]](https://www.zhihu.com/equation?tex=B) 如下（原文是大写的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) ，知乎没这个符号）：

![[公式]](https://www.zhihu.com/equation?tex=B%3AL%27%5ET%5Crightarrow+L%5E%7B%5Cleq+T%7D%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=L%27) 是上述加入blank的长度为 ![[公式]](https://www.zhihu.com/equation?tex=T) 的字符集合，经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换后得到原始 ![[公式]](https://www.zhihu.com/equation?tex=L) ，显然对于![[公式]](https://www.zhihu.com/equation?tex=L)的最大长度有 ![[公式]](https://www.zhihu.com/equation?tex=%7CL%7C%5Cleq+T) 。

举例说明，当 ![[公式]](https://www.zhihu.com/equation?tex=T%3D12) 时：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+B%28%5Cpi_1%29%26%3DB%28--s%5C+t%5C+t%5C+a-t---e%29%3Dstate%5C%5C+%5Cend%7Balign%2A%7D%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=B%28%5Cpi_2%29%3DB%28s%5C+s%5C+t-a%5C+a%5C+a-t%5C+e%5C+e-%29%3Dstate%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=B%28%5Cpi_3%29%3DB%28--s%5C+t%5C+t%5C+a%5C+a-t%5C+e%5C+e-%29%3Dstate%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=B%28%5Cpi_4%29%3DB%28s%5C+s%5C+t-a%5C+a-t---e%29%3Dstate%5C%5C)

对于字符间有blank符号的则不合并：

![[公式]](https://www.zhihu.com/equation?tex=B%28%5Cpi_5%29%3DB%28-%5C+s%5C+t%5C+a-a%5C+t%5C+t%5C+e-%5C+e-%29%3Dstaatee%5C%5C)

当获得LSTM输出![[公式]](https://www.zhihu.com/equation?tex=y)后进行![[公式]](https://www.zhihu.com/equation?tex=B)变换，即可获得输出结果。显然 **![[公式]](https://www.zhihu.com/equation?tex=B)** 变换**不是单对单映射**，例如对于不同的![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%5Csim+%5Cpi_4)都可获得英文单词state。同时 ![[公式]](https://www.zhihu.com/equation?tex=%7CL%7C%3D%7C%7Bstate%7D%7C+%3D+5%5Cleq+12%3DT) 成立。

**那么CTC怎么做？**

对于LSTM给定输入 ![[公式]](https://www.zhihu.com/equation?tex=x) 的情况下，输出为 ![[公式]](https://www.zhihu.com/equation?tex=l) 的概率为：

![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29%3D%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%7D%7B%7Dp%28%5Cpi%7Cx%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29) 代表所有经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换后是 ![[公式]](https://www.zhihu.com/equation?tex=l) 的路径 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) 。

其中，对于任意一条路径 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) 有：

![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi%7Cx%29%3D%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dy_%7B%5Cpi_t%7D%5Et%5C+%2C%5C+%5Cforall+%5Cpi+%5Cin+L%27%5ET%5C%5C)

注意这里的 ![[公式]](https://www.zhihu.com/equation?tex=y_%7B%5Cpi_t%7D%5Et) 中的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_t) ，下标 ![[公式]](https://www.zhihu.com/equation?tex=t) 表示 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) 路径的每一个时刻；而上面 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%5Csim+%5Cpi_4) 的下标表示不同的路径。两个下标含义不同注意区分。

***注意上式 ![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi%7Cx%29%3D%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dy_%7B%5Cpi_t%7D%5Et) 成立有条件，此项不做进一步讨论，有兴趣的读者请自行研究。**

如对于 ![[公式]](https://www.zhihu.com/equation?tex=T%3D12) 的路径 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1) 来说：

![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%3D%28--s%5C+t%5C+t%5C+a-t---e%EF%BC%89%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi_1%7Cx%29%3Dy_-%5E1%5Ccdot+y_-%5E2%5Ccdot+y_s%5E3%5Ccdot+y_t%5E4%5Ccdot+y_t%5E5%5Ccdot+y_a%5E6%5Ccdot+y_-%5E7%5Ccdot+y_t%5E8%5Ccdot+y_-%5E9%5Ccdot+y_-%5E%7B10%7D%5Ccdot+y_-%5E%7B11%7D%5Ccdot+y_e%5E%7B12%7D%5C%5C)

实际情况中一般手工设置 ![[公式]](https://www.zhihu.com/equation?tex=T%5Cgeq20) ，所以有非常多条 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29) 路径，即 ![[公式]](https://www.zhihu.com/equation?tex=%7CB%5E%7B-1%7D%28l%29%7C) 非常大，无法逐条求和直接计算 ![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29) 。所以需要一种快速计算方法。

**CTC的训练目标**

![img](https://pic3.zhimg.com/v2-aa0e26bbce5b7b45ce7b6c767b6584b2_r.jpg)图14

CTC的训练过程，本质上是通过梯度 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+w%7D) 调整LSTM的参数 ![[公式]](https://www.zhihu.com/equation?tex=w) ，使得对于输入样本为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29) 时使得 ![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29) 取得最大。

例如下面图14的训练样本，目标都是使得 ![[公式]](https://www.zhihu.com/equation?tex=l%3Dstate) 时的输出 ![[公式]](https://www.zhihu.com/equation?tex=p%28l%3D%5Ctext%7Bstate%7D%7Cx%29) 变大。

![img](https://pic2.zhimg.com/80/v2-faa81dec02bd2bb4b186808a3b7ac689_720w.jpg)图14

**CTC借用了HMM的“向前—向后”(forward-backward)算法来计算** ![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29)

要计算 ![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29) ，由于有blank的存在，定义路径 ![[公式]](https://www.zhihu.com/equation?tex=l%27) 为在路径 ![[公式]](https://www.zhihu.com/equation?tex=l) 每两个元素以及头尾插入blank。那么对于任意的 ![[公式]](https://www.zhihu.com/equation?tex=l%27_i) 都有 ![[公式]](https://www.zhihu.com/equation?tex=+l%27_i+%5Cin+L%27) （其中 ![[公式]](https://www.zhihu.com/equation?tex=L%27%3DL%5Ccup+%5C%7Bblank%5C%7D) ）。如：

![[公式]](https://www.zhihu.com/equation?tex=l%3Dstate%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=+l%27%3D-s-t-a-t-e-%5C%5C)

显然 ![[公式]](https://www.zhihu.com/equation?tex=%7Cl%27%7C%3D2%7Cl%7C%2B1) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=%7Cl%7C) 是路径的最大长度，如上述例子中 ![[公式]](https://www.zhihu.com/equation?tex=%7Cl%7C%3D%7Cstate%7C%3D5) 。

定义所有经 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换后结果是 ![[公式]](https://www.zhihu.com/equation?tex=l) 且在 ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻结果为 ![[公式]](https://www.zhihu.com/equation?tex=l_k)（记为![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_t%3Dl_k) ）的路径集合为 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%5Cpi%7C%5Cpi%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%5C%7D) 。

求导：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D%26%3D%5Cfrac%7B%5Cpartial+%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%7D%7B%7Dp%28%5Cpi%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D%5C%5C+%26%3D%5Cfrac%7B%5Cpartial+%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7D%7B%7Dp%28%5Cpi%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D%2B%5Cfrac%7B%5Cpartial+%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%5Cne+l_k%7D%7B%7Dp%28%5Cpi%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D+%5Cend%7Balign%7D%5C%5C)

注意上式中第二项与 ![[公式]](https://www.zhihu.com/equation?tex=y_k%5Et) 无关，所以：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D%3D%5Cfrac%7B%5Cpartial+%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7D%7B%7Dp%28%5Cpi%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D)

而上述 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D) 就是恰好与概率 ![[公式]](https://www.zhihu.com/equation?tex=y_k%5Et) 相关的路径，即 ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻都经过 ![[公式]](https://www.zhihu.com/equation?tex=l_k) (![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_t%3Dl_k) )。

举例说明，还是看上面的例子 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%2C%5Cpi_2) （这里的下标 ![[公式]](https://www.zhihu.com/equation?tex=1%2C2) 代表不同的路径）：

![img](https://pic1.zhimg.com/v2-b35c7212d02f2c4847a6038a5ef9a200_r.jpg)图15

蓝色路径 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1) ：

![[公式]](https://www.zhihu.com/equation?tex=B%28%5Cpi_1%29%3DB%28--s%5C+t%5C+t%5C+a-t---e%29%3Dstate%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi_1%7Cx%29%3Dy_-%5E1%5Ccdot+y_-%5E2%5Ccdot+y_s%5E3%5Ccdot+y_t%5E4%5Ccdot+y_t%5E5%5Ccdot+y_a%5E6%5Ccdot+y_-%5E7%5Ccdot+y_t%5E8%5Ccdot+y_-%5E9%5Ccdot+y_-%5E%7B10%7D%5Ccdot+y_-%5E%7B11%7D%5Ccdot+y_e%5E%7B12%7D%5C%5C)

红色路径 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1) ：

![[公式]](https://www.zhihu.com/equation?tex=B%28%5Cpi_2%29%3DB%28s%5C+s%5C+t-a%5C+a%5C+a-t%5C+e%5C+e-%29%3Dstate%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi_2%7Cx%29%3Dy_s%5E1%5Ccdot+y_s%5E2%5Ccdot+y_t%5E3%5Ccdot+y_-%5E4%5Ccdot+y_a%5E5%5Ccdot+y_a%5E6%5Ccdot+y_a%5E7%5Ccdot+y_-%5E8%5Ccdot+y_t%5E9%5Ccdot+y_e%5E%7B10%7D%5Ccdot+y_e%5E%7B11%7D%5Ccdot+y_-%5E%7B12%7D%5C%5C)

还有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_3%2C%5Cpi_4) 没有画出来。

而 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1+%2C%5Cpi_2%2C%5Cpi_3%2C%5Cpi_4) 在 ![[公式]](https://www.zhihu.com/equation?tex=t%3D6) 时恰好都经过 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_6%3Da) （此处下标代表路径 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) 的 ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻的字符）。所有类似于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1+%2C%5Cpi_2%2C%5Cpi_3%2C%5Cpi_4) 经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换后结果是 ![[公式]](https://www.zhihu.com/equation?tex=l%3Dstate) 且在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_6%3Da) 的路径集合表示为 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%5Cpi%7C%5Cpi%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_6%3Da%5C%7D) 。

观察 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1+%2C%5Cpi_2%2C%5Cpi_3%2C%5Cpi_4) 。记 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1) 蓝色为 ![[公式]](https://www.zhihu.com/equation?tex=b%28blue%29) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_2) 红色路径为 ![[公式]](https://www.zhihu.com/equation?tex=r%28red%29) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%2C%5Cpi_2) 可以表示：

![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%3Db%3Db_%7B1%3A5%7D%2Ba_6%2Bb_%7B7%3A12%7D%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_2%3Dr%3Dr_%7B1%3A5%7D%2Ba_6%2Br_%7B7%3A12%7D%5C%5C)

那么 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_3%2C%5Cpi_4) 可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_3%3Db_%7B1%3A5%7D%2Ba_6%2Br_%7B7%3A12%7D%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_4%3Dr_%7B1%3A5%7D%2Ba_6%2Bb_%7B7%3A12%7D%5C%5C)

计算：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+y_a%5E6%7D%26%3D%5Cfrac%7B%5Cpartial+%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_6%3Da%7D%7B%7Dp%28%5Cpi%7Cx%29%7D%7B%5Cpartial+y_a%5E6%7D%5C%5C%26%3D%5Cfrac%7B%5Cpartial+p%28%5Cpi_1%7Cx%29%2B%5Cpartial+p%28%5Cpi_2%7Cx%29%2B%5Cpartial+p%28%5Cpi_3%7Cx%29%2B%5Cpartial+p%28%5Cpi_4%7Cx%29%2B...%7D%7B%5Cpartial+y_a%5E6%7D%5C%5C+%5Cend%7Balign%7D%5C%5C)

为了观察规律，单独计算 ![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi_1%7Cx%29%2Bp%28%5Cpi_2%7Cx%29%2Bp%28%5Cpi_3%7Cx%29%2Bp%28%5Cpi_4%7Cx%29) 。

![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi_1%7Cx%29%2Bp%28%5Cpi_2%7Cx%29%2Bp%28%5Cpi_3%7Cx%29%2Bp%28%5Cpi_4%7Cx%29)

![[公式]](https://www.zhihu.com/equation?tex=%3Dy_-%5E1%5Ccdot+y_-%5E2%5Ccdot+y_s%5E3%5Ccdot+y_t%5E4%5Ccdot+y_t%5E5%5Ccdot+y_a%5E6%5Ccdot+y_-%5E7%5Ccdot+y_t%5E8%5Ccdot+y_-%5E9%5Ccdot+y_-%5E%7B10%7D%5Ccdot+y_-%5E%7B11%7D%5Ccdot+y_e%5E%7B12%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5C+%5C+%2By_s%5E1%5Ccdot+y_s%5E2%5Ccdot+y_t%5E3%5Ccdot+y_-%5E4%5Ccdot+y_a%5E5%5Ccdot+y_a%5E6%5Ccdot+y_a%5E7%5Ccdot+y_-%5E8%5Ccdot+y_t%5E9%5Ccdot+y_e%5E%7B10%7D%5Ccdot+y_e%5E%7B11%7D%5Ccdot+y_-%5E%7B12%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5C+%5C+%2By_-%5E1%5Ccdot+y_-%5E2%5Ccdot+y_s%5E3%5Ccdot+y_t%5E4%5Ccdot+y_t%5E5%5Ccdot+y_a%5E6%5Ccdot+y_a%5E7%5Ccdot+y_-%5E8%5Ccdot+y_t%5E9%5Ccdot+y_e%5E%7B10%7D%5Ccdot+y_e%5E%7B11%7D%5Ccdot+y_-%5E%7B12%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5C+%5C+%2By_s%5E1%5Ccdot+y_s%5E2%5Ccdot+y_t%5E3%5Ccdot+y_-%5E4%5Ccdot+y_a%5E5%5Ccdot+y_a%5E6%5Ccdot+y_-%5E7%5Ccdot+y_t%5E8%5Ccdot+y_-%5E9%5Ccdot+y_-%5E%7B10%7D%5Ccdot+y_-%5E%7B11%7D%5Ccdot+y_e%5E%7B12%7D)

不妨令：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctexttt%7Bforward%7D%3Dp%28b_%7B1%3A5%7D%2Br_%7B1%3A5%7D%7Cx%29%3Dy_-%5E1%5Ccdot+y_-%5E2%5Ccdot+y_s%5E3%5Ccdot+y_t%5E4%5Ccdot+y_t%5E5%2By_s%5E1%5Ccdot+y_s%5E2%5Ccdot+y_t%5E3%5Ccdot+y_-%5E4%5Ccdot+y_a%5E5%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctexttt%7Bbackward%7D%3Dp%28b_%7B7%3A12%7D%2Br_%7B7%3A12%7D%7Cx%29%3Dy_-%5E7%5Ccdot+y_t%5E8%5Ccdot+y_-%5E9%5Ccdot+y_-%5E%7B10%7D%5Ccdot+y_-%5E%7B11%7D%5Ccdot+y_e%5E%7B12%7D%2By_a%5E7%5Ccdot+y_-%5E8%5Ccdot+y_t%5E9%5Ccdot+y_e%5E%7B10%7D%5Ccdot+y_e%5E%7B11%7D%5Ccdot+y_-%5E%7B12%7D%5C%5C)

那么![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi_1%7Cx%29%2Bp%28%5Cpi_2%7Cx%29%2Bp%28%5Cpi_3%7Cx%29%2Bp%28%5Cpi_4%7Cx%29)可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi_1%7Cx%29%2Bp%28%5Cpi_2%7Cx%29%2Bp%28%5Cpi_3%7Cx%29%2Bp%28%5Cpi_4%7Cx%29%3D%7B%5Ctexttt%7Bforward%7D%7D+%5Ccdot+y_a%5Et+%5Ccdot+%7B%5Ctexttt%7Bbackward%7D%7D%5C%5C)

推广一下，所有经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换为 ![[公式]](https://www.zhihu.com/equation?tex=l) 且 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_6%3Da) 的路径（即 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%5Cpi%7C%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_6%3Da%5C%7D) ）可以写成如下形式：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_6%3Da%7Dp%28%5Cpi%7Cx%29%3D%7B%5Ctexttt%7Bforward%7D%7D+%5Ccdot+y_a%5Et+%5Ccdot+%7B%5Ctexttt%7Bbackward%7D%7D%5C%5C)

进一步推广，所有经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换为 ![[公式]](https://www.zhihu.com/equation?tex=l) 且 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_t%3Dl_k) 的路径（即 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%5Cpi%7C%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%5C%7D) ）也都可以写作：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7Dp%28%5Cpi%7Cx%29%3D%7B%5Ctexttt%7Bforward%7D%7D+%5Ccdot+y_%7Bl_k%7D%5Et+%5Ccdot+%7B%5Ctexttt%7Bbackward%7D%7D%5C%5C)

**所以，定义前向递推概率和** ![[公式]](https://www.zhihu.com/equation?tex=%5Ctexttt%7Bforward%7D%3D%5Calpha_t%28s%29) **：**

对于一个长度为 ![[公式]](https://www.zhihu.com/equation?tex=T) 的路径 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_%7B1%3At%7D) 代表该路径前 ![[公式]](https://www.zhihu.com/equation?tex=t) 个字符， ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_%7Bt%3AT%7D) 代表后 ![[公式]](https://www.zhihu.com/equation?tex=T-t) 个字符。

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28s%29%3D%5Csum_%7B%5Cpi+%5Cin+B%28%5Cpi_%7B1%3At%7D%29%3Dl_%7B1%3As%7D%7D%7B%5Cprod_%7Bt%27%3D1%7D%5E%7Bt%7Dy_%7B%5Cpi_%7Bt%27%7D%7D%5E%7Bt%27%7D%7D%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%28%5Cpi_%7B1%3At%7D%29%3Dl_%7B1%3As%7D) 表示前 ![[公式]](https://www.zhihu.com/equation?tex=t) 个字符 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_%7B1%3At%7D) 经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换为的 ![[公式]](https://www.zhihu.com/equation?tex=l_%7B1%3As%7D) 的前半段子路径。 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28s%29) 代表了 ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻经过 ![[公式]](https://www.zhihu.com/equation?tex=l_s) 的路径概率中 ![[公式]](https://www.zhihu.com/equation?tex=1%5Csim+t) 概率之和，即前向递推概率和。

由于当 ![[公式]](https://www.zhihu.com/equation?tex=t%3D1) 时路径只能从blank或 ![[公式]](https://www.zhihu.com/equation?tex=l_1) 开始，所以 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28s%29) 有如下性质：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_1%28-%29%3Dy_-%5E1%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_1%28l_1%29%3Dy_%7Bl_1%7D%5E1%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_1%28l_t%29%3D0%5C+%2C%5C+%5Cforall+t%3E1%5C%5C)

如上面的例子中 ![[公式]](https://www.zhihu.com/equation?tex=T%3D12) , ![[公式]](https://www.zhihu.com/equation?tex=l%3Dstate) , ![[公式]](https://www.zhihu.com/equation?tex=l_%7B1%3A3%7D%3Dsta) 。对于所有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%28%5Cpi_%7B1%3A6%7D%29%3Dl_%7B1%3A3%7D) 路径，当 ![[公式]](https://www.zhihu.com/equation?tex=t%3D1) 时只能从blank和 ![[公式]](https://www.zhihu.com/equation?tex=s) 字符开始。

![img](https://pic2.zhimg.com/v2-2371b50895b935ecb8f3925a0462e5e5_r.jpg)图16

图16是 ![[公式]](https://www.zhihu.com/equation?tex=T%3D12) 时经过压缩路径后能够变为 ![[公式]](https://www.zhihu.com/equation?tex=l%3Dstate) 的所有路径 ![[公式]](https://www.zhihu.com/equation?tex=B%5E%7B-1%7D%28l%29) 。观察图15会发现对于 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_6%28a%29) 有如下递推关系：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_6%28a%29%3D%28%5Calpha_5%28a%29%2B%5Calpha_5%28t%29%2B%5Calpha_5%28-%29%29%5Ccdot+y_a%5E6%5C%5C)

也就是说，如果 ![[公式]](https://www.zhihu.com/equation?tex=t%3D6) 时刻是字符 ![[公式]](https://www.zhihu.com/equation?tex=a) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=t%3D5) 时刻只可能是字符 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ct%2Cblank) 三选一，否则经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换后无法压缩成 ![[公式]](https://www.zhihu.com/equation?tex=l%3Dstate) 。

那么更一般的：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28l%27_k%29%3D%28%5Calpha_%7Bt-1%7D%28l%27_k%29%2B%5Calpha_%7Bt-1%7D%28l%27_%7Bk-1%7D%29%2B%5Calpha_%7Bt-1%7D%28-%29%29%5Ccdot+y_%7Bl%27_k%7D%5Et%5C%5C)

**同理，定义反向递推概率和** ![[公式]](https://www.zhihu.com/equation?tex=%5Ctexttt%7Bbackword+%7D%3D%5Cbeta_t%28s%29) **：**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_t%28s%29%3D%5Csum_%7B%5Cpi+%5Cin+B%28%5Cpi_%7Bt%3AT%7D%29%3Dl_%7Bs%3A%7Cl%7C%7D%7D%7B%5Cprod_%7Bt%27%3Dt%7D%5E%7BT%7Dy_%7B%5Cpi_%7Bt%27%7D%7D%5E%7Bt%27%7D%7D%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%28%5Cpi_%7Bt%3AT%7D%29%3Dl_%7Bs%3A%7Cl%7C%7D) 表示后 ![[公式]](https://www.zhihu.com/equation?tex=T-t) 个字符 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_%7Bt%3AT%7D) 经过 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换为的 ![[公式]](https://www.zhihu.com/equation?tex=l_%7Bs%3A%7Cl%7C%7D) 的后半段子路径。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_t%28s%29) 代表了 ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻经过 ![[公式]](https://www.zhihu.com/equation?tex=l_s) 的路径概率中 ![[公式]](https://www.zhihu.com/equation?tex=t%5Csim+T) 概率之和，即反向递推概率和。

由于当 ![[公式]](https://www.zhihu.com/equation?tex=t%3DT) 时路径只能以blank或 ![[公式]](https://www.zhihu.com/equation?tex=l_%7B%7Cl%27%7C%7D) 结束，所以有如下性质：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_T%28-%29%3Dy_-%5ET%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_T%28l%27_%7B%7Cl%27%7C%7D%29%3Dy_%7Bl%27_%7B%7Cl%27%7C%7D%7D%5ET%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_T%28l%27_%7B%7Cl%27%7C-i%7D%29%3D0%5C+%2C%5C+%5Cforall+i%3E0%5C%5C)

如上面的例子中 ![[公式]](https://www.zhihu.com/equation?tex=T%3D12) , ![[公式]](https://www.zhihu.com/equation?tex=l%3Dstate) , ![[公式]](https://www.zhihu.com/equation?tex=%7Cl%7C%3D%7Cstate%7C%3D5) , ![[公式]](https://www.zhihu.com/equation?tex=l_%7B3%3A5%7D%3Date) 。对于所有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%28%5Cpi_%7B6%3A12%7D%29%3Dl_%7B3%3A6%7D) 路径，当 ![[公式]](https://www.zhihu.com/equation?tex=t%3D12) 时只能以 ![[公式]](https://www.zhihu.com/equation?tex=l%27_%7B%7Cl%27%7C%7D%3Dl%27_%7B11%7D%3D-) （blank字符）或 ![[公式]](https://www.zhihu.com/equation?tex=l%27_%7B%7Cl%27%7C%7D%3Dl%27_%7B11%7D%3De) 字符结束。

观察图15会发现对于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_6%28a%29) 有如下递推关系

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_6%28a%29%3D+%28%5Cbeta_7%28a%29+%2B+%5Cbeta_7%28t%29+%2B+%5Cbeta_7%28-%29%29+%5Ccdot+y_a%5E6++%5C%5C)

与 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28s%29) 同理，对于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_t%28s%29) 有如下递推关系：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_t%28l%27_k%29%3D%28%5Cbeta_%7Bt%2B1%7D%28l%27_k%29%2B%5Cbeta_%7Bt%2B1%7D%28l%27_%7Bk%2B1%7D%29%2B%5Cbeta_%7Bt%2B1%7D%28-%29%29%5Ccdot+y_%7Bl%27_k%7D%5Et%5C%5C)

那么forward和backward相乘有：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%5Calpha_t%28l%27_k%29%5Cbeta_t%28l%27_k%29+%26%3D%5Csum_%7B%5Cpi%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl%27_k%7Dy_%7Bl%27_k%7D%5Et%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dy_%7B%5Cpi_t%7D%5E%7Bt%7D%5C%5C++%5Cend%7Balign%2A%7D%5C%5C)

或：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28l_k%29%5Cbeta_t%28l_k%29%3D%5Csum_%7B%5Cpi%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7Dy_%7Bl_k%7D%5Et%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dy_%7B%5Cpi_t%7D%5E%7Bt%7D%5C%5C)

注意， ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bl_k%7D%5Et%E4%B8%8Ey_%7Bl%E2%80%99_k%7D%5Et) 可以通过图16的关系对应，如 ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bl_1%7D%5Et%3Dy_%7Bl%E2%80%99_2%7D%5Et) ，![[公式]](https://www.zhihu.com/equation?tex=y_%7Bl_2%7D%5Et%3Dy_%7Bl%E2%80%99_4%7D%5Et)。

对比 ![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29) :

![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29%3D%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%7D%7B%7Dp%28%5Cpi%7Cx%29%3D%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%7D%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dy_%7B%5Cpi_t%7D%5Et%5C%5C)

可以得到 ![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29) 与forward和backward递推公式之间的关系：

![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29+%3D+%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7D%7B%7D%5Cfrac%7B%5Calpha_t%28l_k%29%5Cbeta_t%28l_k%29%7D%7By_%7Bl_k%7D%5Et%7D%5C%5C)



*** 为什么有上式 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28l_k%29%5Cbeta_t%28l_k%29%3D%5Csum_%7B%5Cpi%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7Dy_%7Bl_k%7D%5Et%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dy_%7B%5Cpi_t%7D%5E%7Bt%7D) 成立呢？**

回到图15，为了方便分析，假设只有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%5Csim%5Cpi_4) 共4条在 ![[公式]](https://www.zhihu.com/equation?tex=t%3D6) 时刻经过字符 ![[公式]](https://www.zhihu.com/equation?tex=a) 且 ![[公式]](https://www.zhihu.com/equation?tex=B) 变换为 ![[公式]](https://www.zhihu.com/equation?tex=l) 的路径，即 :

![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1%3Db_%7B1%3A5%7D%2Ba_6%2Bb_%7B7%3A12%7D%5C%5C+%5Cpi_2%3Dr_%7B1%3A5%7D%2Ba_6%2Br_%7B7%3A12%7D%5C%5C+%5Cpi_3%3Db_%7B1%3A5%7D%2Ba_6%2Br_%7B7%3A12%7D%5C%5C+%5Cpi_4%3Dr_%7B1%3A5%7D%2Ba_6%2Bb_%7B7%3A12%7D%5C%5C)

那么此时（注意虽然表示路径用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi%3D%5Cpi_1%2B%5Cpi_2) 加法，但是由于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_2) 两件独立事情同时发生，所以 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) 路径的概率 ![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cpi%7Cx%29%3Dp%28%5Cpi_1%7Cx%29p%28%5Cpi_2%7Cx%29) 是乘法）：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28l_k%27%29%3Dp%28b_%7B1%3A5%7D%7Cx%29%5Ccdot+y_%7Bl%27_k%7D%5E6%2Bp%28r_%7B1%3A5%7D%7Cx%29%5Ccdot+y_%7Bl%27_k%7D%5E6%5C%5C+%5Cbeta_t%28l_k%27%29%3Dp%28b_%7B7%3A12%7D%7Cx%29%5Ccdot+y_%7Bl%27_k%7D%5E6%2Bp%28r_%7B7%3A12%7D%7Cx%29%5Ccdot+y_%7Bl%27_k%7D%5E6%5C%5C)

则有：

![img](https://pic2.zhimg.com/v2-ef2eaf1c36fe5af6a0e0e1a0f4cc4955_r.jpg)

**训练CTC**

对于LSTM，有训练集合 ![[公式]](https://www.zhihu.com/equation?tex=S%3D%5C%7B%28x_1%2Cz_1%29%2C%28x_1%2Cz_1%29%2C...%2C%28x_N%2Cz_N%29%5C%7D) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=x) 是图片经过CNN计算获得的Feature map， ![[公式]](https://www.zhihu.com/equation?tex=z) 是图片对应的OCR字符label（label里面没有blank字符）。

**现在我们要做的事情就是**：通过梯度![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+w%7D)调整LSTM的参数![[公式]](https://www.zhihu.com/equation?tex=w)，使得对于输入样本为![[公式]](https://www.zhihu.com/equation?tex=%5Cpi+%5Cin+B%5E%7B-1%7D%28z%29)时有 ![[公式]](https://www.zhihu.com/equation?tex=p%28l%7Cx%29) 取得最大。**所以如何计算梯度才是核心。**

单独来看CTC输入（即LSTM输出） ![[公式]](https://www.zhihu.com/equation?tex=y) 矩阵中的某一个值 ![[公式]](https://www.zhihu.com/equation?tex=y_k%5Et) （注意 ![[公式]](https://www.zhihu.com/equation?tex=y_k%5Et) 与 ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bl_k%7D%5Et) 含义相同，都是在 ![[公式]](https://www.zhihu.com/equation?tex=t) 时 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_t%3Dl_k) 的概率）：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D%26%3D%5Cfrac%7B%5Cpartial+%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7D%7B%7D%5Cfrac%7B%5Calpha_t%28l_k%29%5Cbeta_t%28l_k%29%7D%7By_%7Bl_k%7D%5Et%7D%7D%7B%5Cpartial+y_%7Bl_k%7D%5Et%7D%5C%5C%26%3D-%5Cfrac%7B%5Csum_%7B%5Cpi+%5Cin+B%5E%7B-1%7D%28l%29%2C%5Cpi_t%3Dl_k%7D%5Calpha_t%28l_k%29%5Cbeta_t%28l_k%29%7D%7B%28y_%7Bl_k%7D%5Et%29%5E2%7D+%5Cend%7Balign%7D%5C%5C)

上式中的 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_t%28l_k%29%5Cbeta_t%28l_k%29) 是通过递推计算的常数，任何时候都可以通过递推快速获得，那么即可快速计算梯度 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p%28l%7Cx%29%7D%7B%5Cpartial+y_k%5Et%7D) ，之后梯度上升算法你懂的。

**CTC编程接口**

在Tensorflow中官方实现了CTC接口：

```python3
tf.nn.ctc_loss(
    labels,
    inputs,
    sequence_length,
    preprocess_collapse_repeated=False,
    ctc_merge_repeated=True,
    ignore_longer_outputs_than_inputs=False,
    time_major=True
)
```

在Pytorch中需要使用针对框架编译的warp-ctc：[https://github.com/SeanNaren/warp-ctc](https://link.zhihu.com/?target=https%3A//github.com/SeanNaren/warp-ctc)

2020.4更新，目前Pytorch已经有CTC接口：

```text
torch.nn.CTCLoss(blank=0,reduction='mean',zero_infinity=False）
```

**CTC总结**

CTC是一种Loss计算方法，用CTC代替Softmax Loss，训练样本无需对齐。CTC特点：

- 引入blank字符，解决有些位置没有字符的问题
- 通过递推，快速计算梯度

看到这里你也应该大致了解MFCC+CTC在语音识别中的应用了（[图17来源](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_39012047/article/details/83864031)）。

![img](https://pic4.zhimg.com/v2-31abfbb4be0cc6995367ea956488c20b_r.jpg)图17 MFCC+CTC在语音识别中的应用

## CRNN+CTC总结

这篇文章的核心，就是将CNN/LSTM/CTC三种方法结合：

- 首先CNN提取图像卷积特征
- 然后LSTM进一步提取图像卷积特征中的序列特征
- 最后引入CTC解决训练时字符无法对齐的问题

即提供了一种end2end文字图片识别算法，也算是方向的简单入门。

## 特别说明

一般情况下对一张图像中的文字进行识别需要以下步骤

1. 定位文稿中的图片，表格，文字区域，区分文字段落（版面分析）
2. 进行文本行识别（识别）
3. 使用NLP相关算法对文字识别结果进行矫正（后处理）

本文介绍的CRNN框架只是步骤2的一种识别算法，其他非本文内容。CTC你学会(fei)了么？



转载自：https://zhuanlan.zhihu.com/p/43534801