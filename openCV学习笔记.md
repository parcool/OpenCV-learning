[TOC]



#### 图像与像素之间的关系

<img src="http://img.yibogame.com/uPic/2022-04-20-15-52-32-1650441152444-lena_img.png" alt="lena_img" style="zoom:75%;" />

#### 数据图像格式

`cv2.IMREAD_COLOR`: 彩色图像
`cv2.IMREAD_GRAYSCALE`: 灰度图像

#### 读取图像方法
`img = cv2.imread('cat.jpg')`

#### 输出图像
`img`

```
[out]:
#[h,w,c]
array([
[[100,100,100],[200,200,200],[300,300,300]…,[900,900,900]],
[[111,111,111],[222,222,222],[333,333,333]…,[999,999,999]],
[[123,123,123],[234,234,234],[345,345,345]…,[789,789,789]]
],dtype=unit8)
```

#### 显示图像
```python
cv2.imshow('image_name',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 图像数据大小与格式
`img.shape`
```
[out]:
(414,500,3)
#也就是上面的[h,w,c]，c就是代表BGR
```

#### 读取为灰度图
`img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)`
`img`

```
[out]:
array([
[100,200,300,…,900],
[111,222,333,…,999],
…
[123,456,789,…,012]
],dtype=unit8)
```

#### 保存图像
`cv2.imwrite('mycat.png',img)`

#### 查看图像的格式
`type(img)`
```
[out]:numpy.ndarray
```
#### 查看图像的大小
`img.size`
```
[out]:
207000
```
#### 查看图像的dtype
`img.dtype`
```
[out]:
dype('unit8')
```

#### 读取视频
cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备，例如0,1。
如果是视频文件，直接指定好路径即可。
```python
vc = cv2.VideoCapture('test.mp4')
# 检查是否打开正确
if vc.isOpened(): 
    open, frame = vc.read()
else:
    open = False
#%%
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(100) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
```

#### 截取部分图像数据
```
img = cv2.imread('cat.jpg`)
cat = img[0:200,0:200]
cv_show('cat',cat)
```

#### 颜色通道提取
`b,g,r = cv2.split(img)`
`b`

```
[out]:
array([[160, 164, 169, ..., 185, 184, 183],
       [126, 131, 136, ..., 184, 183, 182],
       [127, 131, 137, ..., 183, 182, 181],
       ...,
       [198, 193, 178, ..., 206, 195, 174],
       [176, 183, 175, ..., 188, 144, 125],
       [190, 190, 157, ..., 200, 145, 144]], dtype=uint8)
```

#### 颜色的合并
`img = cv2.merge((b,g,r))`

#### 只保留R(其余通道置为0)
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,1] = 0
cv_show('R',cur_img)


#### 边界填充(周围扩大一圈)
```python
# 先定义一下填充的大小
top_size,bottom_size,left_size,right_size = (50,50,50,50)
# 不同的填充方法
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
```
<img src="http://img.yibogame.com/uPic/2022-04-18-21-54-56-1650290096178-2022-04-18-18-58-55-1650279535945-output.png" alt="image-20220418205212143" />
`BORDER_REPLICATE`：复制法，也就是复制最边缘像素。
`BORDER_REFLECT`：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
`BORDER_REFLECT_101`：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
`BORDER_WRAP`：外包装法cdefgh|abcdefgh|abcdefg
`BORDER_CONSTANT`：常量法，常数值填充。

#### 数值计算
```python
img_cat = cv2.imread('cat.jpg')
img_dog = cv2.imread('dog.jpg')
```
```python
img_cat[:5,:,0]
```
```
[out]:
array([[142, 146, 151, ..., 156, 155, 154],
       [107, 112, 117, ..., 155, 154, 153],
       [108, 112, 118, ..., 154, 153, 152],
       [139, 143, 148, ..., 156, 155, 154],
       [153, 158, 163, ..., 160, 159, 158]], dtype=uint8)
```
```python
img_cat2= img_cat +10 
img_cat[:5,:,0]
```
```
[out]:
array([[152, 156, 161, ..., 166, 165, 164],
       [117, 122, 127, ..., 165, 164, 163],
       [118, 122, 128, ..., 164, 163, 162],
       [149, 153, 158, ..., 166, 165, 164],
       [163, 168, 173, ..., 170, 169, 168]], dtype=uint8)
```
```python
#相当于 %(取余) 256
(img_cat + img_cat2)[:5,:,0] 
```
```
[out]:
array([[ 38,  46,  56, ...,  66,  64,  62],
       [224, 234, 244, ...,  64,  62,  60],
       [226, 234, 246, ...,  62,  60,  58],
       [ 32,  40,  50, ...,  66,  64,  62],
       [ 60,  70,  80, ...,  74,  72,  70]], dtype=uint8)
```
```python
cv2.add(img_cat,img_cat2)[:5,:,0]
```
```
[out]:
array([[255, 255, 255, ..., 255, 255, 255],
       [224, 234, 244, ..., 255, 255, 255],
       [226, 234, 246, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255]], dtype=uint8)
```

#### 图像融合
`img_cat + img_dog`
```
[out]:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-111-ffa3cdc5d6b8> in <module>()
----> 1 img_cat + img_dog

ValueError: operands could not be broadcast together with shapes (414,500,3) (429,499,3)
```
因为两个大小不一样，报错了！那先看看`img_cat`的大小：
`img_cat.shape`

```
[out]:
(414, 500, 3)
```
把狗变为猫那么大：
```
img_dog = cv2.resize(img_dog, (500, 414))
img_dog.shape
```
```
[out]:
(414, 500, 3)
```

```python
# Result = αx1 + βx2 + b（x1:猫，x2:狗，b：偏移量，α、β：权重）
res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)
plt.imshow(res)
```
<img src="http://img.yibogame.com/uPic/2022-04-18-18-58-49-1650279529590-output1.png" alt="image-20220418205212143" />

#### 图像阈值处理
`ret, dst = cv2.threshold(src, thresh, maxval, type)`
`src`： 输入图，只能输入单通道图像，通常来说为灰度图
`dst`： 输出图
`thresh`： 阈值
`maxval`： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
`type`：二值化操作的类型，包含以下5种类型： 

`cv2.THRESH_TOZERO_INV`
`cv2.THRESH_BINARY`           超过阈值(也就是`thresh`)就取`maxval`（最大值），否则取`0`
`cv2.THRESH_BINARY_INV`是 `THRESH_BINARY`的反转
`cv2.THRESH_TRUNC`大于阈值部分设为阈值，否则不变
`cv2.THRESH_TOZERO`大于阈值部分不改变，否则设为`0`
`cv2.THRESH_TOZERO_INV`是`THRESH_TOZERO`的反转

```python
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
```

<img src="http://img.yibogame.com/uPic/2022-04-18-18-58-44-1650279524214-output3.png" alt="image-20220418205212143" />

#### 图像平滑处理

<img src="http://img.yibogame.com/uPic/2022-04-18-18-58-37-1650279517969-input1.png" alt="input1" style="zoom:50%;" />

```python
img = cv2.imread('lenaNoise.png')
```
原始带噪点图像：
<img src="http://img.yibogame.com/uPic/2022-04-18-20-49-04-1650286144073-lenaNoise.png" alt="lenaNoise" style="zoom:50%;" />

* 均值滤波器
```python
# 均值滤波(121+75+78+……+235)/9
# 简单的平均卷积操作
blur = cv2.blur(img, (3, 3))
cv2.imshow('blur', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
均值滤波输出的结果：
<img src="http://img.yibogame.com/uPic/2022-04-18-20-52-12-1650286332261-image-20220418205212143.png" alt="image-20220418205212143" style="zoom: 25%;" />
* 方框滤波器
```python
# 方框滤波
# 基本和均值一样，可以选择归一化(与上图一样)
box = cv2.boxFilter(img,-1,(3,3), normalize=True)  
cv2.imshow('box', box)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
方框滤波输出的结果（归一化）：
<img src="http://img.yibogame.com/uPic/2022-04-18-20-52-12-1650286332261-image-20220418205212143.png" alt="image-20220418205212143" style="zoom: 25%;" />

```python
# 方框滤波
# 基本和均值一样，可以选择归一化,容易越界(超过255。当超过255后就是255)
box = cv2.boxFilter(img,-1,(3,3), normalize=False)  
cv2.imshow('box', box)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

方框滤波输出的结果（不归一化）：
<img src="http://img.yibogame.com/uPic/2022-04-18-20-55-19-1650286519100-image-20220418205518999.png" alt="image-20220418205518999" style="zoom: 25%;" />

* 高斯滤波器
```python
# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
aussian = cv2.GaussianBlur(img, (5, 5), 1)  

cv2.imshow('aussian', aussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

高斯滤波的函数图像：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-04-07-1650287047640-image-20220418210407535.png" alt="image-20220418210407535" style="zoom:50%;" />

高斯滤波的卷积核大概类似于这样（越中间影响越大）：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-01-58-1650286918084-image-20220418210157986.png" alt="image-20220418210157986" style="zoom:50%;" />

高斯滤波输出的结果：
<img src="http://img.yibogame.com/uPic/2022-04-18-20-59-36-1650286776559-image-20220418205936456.png" alt="image-20220418205936456" style="zoom:25%;" />

* 中值滤波器
```python
# 中值滤波（把121,75,78，……，235按数值大小排序，再取中间的那个值）
# 相当于用中值代替
median = cv2.medianBlur(img, 5)  # 中值滤波

cv2.imshow('median', median)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
中值滤波输出的结果：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-09-00-1650287340845-image-20220418210900722.png" alt="image-20220418210900722" style="zoom:25%;" />

```python
# 展示所有的
res = np.hstack((blur,aussian,median))
# 垂直展示也行：
# res = np.vstack((blur,aussian,median))
cv2.imshow('median vs average', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
`hstack`方式输出结果：

<img src="http://img.yibogame.com/uPic/2022-04-18-21-16-04-1650287764298-image-20220418211604185.png" alt="image-20220418211604185" style="zoom:25%;" />


#### 形态学----腐蚀(erode)
```python
img = cv2.imread('res/dige.png')

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
原始图像：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-25-15-1650288315808-image-20220418212515704.png" alt="image-20220418212515704" style="zoom:35%;" />

```python
#3*3的卷积核
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

cv2.imshow('erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
腐蚀后的图像：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-25-37-1650288337151-image-20220418212537049.png" alt="image-20220418212537049" style="zoom:35%;" />

```python
pie = cv2.imread('res/pie.png')

cv2.imshow('pie', pie)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
腐蚀原理----卷积核（只要里面非全为1就腐蚀掉）：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-38-46-1650289126077-image-20220418213845958.png" alt="image-20220418213845958" style="zoom:35%;" />

```python
#30*30的卷积核
kernel = np.ones((30, 30), np.uint8)
#腐蚀1、2、3次
erosion_1 = cv2.erode(pie, kernel, iterations=1)
erosion_2 = cv2.erode(pie, kernel, iterations=2)
erosion_3 = cv2.erode(pie, kernel, iterations=3)
res = np.hstack((erosion_1, erosion_2, erosion_3))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
腐蚀1、2、3次的结果图：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-39-49-1650289189888-image-20220418213949773.png" alt="image-20220418213949773" style="zoom:35%;" />

#### 形态学----膨胀(dilate)

膨胀与腐蚀互为逆运算

```python
img = cv2.imread('res/dige.png')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
原始图像：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-25-15-1650288315808-image-20220418212515704.png" alt="image-20220418212515704" style="zoom:35%;" />

```python
# 先腐蚀一次
kernel = np.ones((3, 3), np.uint8)
dige_erosion = cv2.erode(img, kernel, iterations=1)

cv2.imshow('dige_erosion', dige_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
腐蚀后输出图像（减肥）：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-25-37-1650288337151-image-20220418212537049.png" alt="image-20220418212537049" style="zoom:35%;" />


```python
# 再膨胀一次
kernel = np.ones((3, 3), np.uint8)
dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)

cv2.imshow('dilate', dige_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
膨胀后输出图像（增肥）：
<img src="http://img.yibogame.com/uPic/2022-04-18-21-49-44-1650289784890-image-20220418214944784.png" alt="image-20220418214944784" style="zoom:35%;" />

```python
# 原图就不展示了，就是上面那个圆
pie = cv2.imread('res/pie.png')

kernel = np.ones((30, 30), np.uint8)
dilate_1 = cv2.dilate(pie, kernel, iterations=1)
dilate_2 = cv2.dilate(pie, kernel, iterations=2)
dilate_3 = cv2.dilate(pie, kernel, iterations=3)
res = np.hstack((dilate_1, dilate_2, dilate_3))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img src="http://img.yibogame.com/uPic/2022-04-18-21-52-02-1650289922866-image-20220418215202759.png" alt="image-20220418215202759" style="zoom:50%;" />

#### 开运算与闭运算（morphology   [mɔːˈfɒlədʒi]  n. （生物）形态学）
**开运算：先腐蚀，再膨胀**

```python
img = cv2.imread('res/dige.png')

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
开运算执行后结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-09-52-10-1650419530695-image-20220420095210587.png" alt="image-20220420095210587" style="zoom:35%;" />

**闭运算：先膨胀，再腐蚀**

```python
img = cv2.imread('res/dige.png')

kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
闭运算执行后结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-09-51-35-1650419495231-image-20220420095135131.png" alt="image-20220420095135131" style="zoom:35%;" />

#### 梯度运算（可以用作取轮廓）
**公式：梯度 = 膨胀 - 腐蚀**。注：这个－是图像学中的减法

```python
pie = cv2.imread('res/pie.png')
kernel = np.ones((7, 7), np.uint8)

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
梯度运算执行后结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-09-59-35-1650419975865-image-20220420095935762.png" alt="image-20220420095935762" style="zoom:35%;" />

#### 礼帽与黑帽
* 礼帽(TopHat)：
**公式：礼帽 = 原始输入 - 开运算结果**
```python
img = cv2.imread('res/dige.png')
kernel = np.ones((7, 7), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
礼帽运算执行后结果（只剩下刺了）：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-05-24-1650420324275-image-20220420100524171.png" alt="image-20220420100524171" style="zoom:35%;" />

* 黑帽(BlackHat)：
**公式：黑猫 = 闭运算 - 原始输入**

```python
img = cv2.imread('res/dige.png')
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat ', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
黑帽运算执行后结果（只剩下原始模糊的轮廓）：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-06-30-1650420390785-image-20220420100630682.png" alt="image-20220420100630682" style="zoom:35%;" />

#### 图像梯度----Sobel算子
公式：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-13-59-1650420839900-sobel_1.png" alt="sobel_1" style="zoom:50%;" />
Gx：（Gradient x）既x（水平）方向上的梯度
计算参考图（左 - 右）：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-23-27-1650421407298-image-20220420102327190.png" alt="image-20220420102327190" style="zoom:35%;" />

Gy：（Gradient y）既y（垂直）方向上的梯度
计算参考图略，因与Gx同理(下 - 上)

`dst = cv2.Sobel(src, ddepth, dx, dy, ksize)`
`ddepth`：图像的深度
`dx`和`dy`：分别表示水平和竖直方向
`ksize`：`Sobel`算子的大小



```python
img = cv2.imread('res/pie.png', cv2.IMREAD_GRAYSCALE)
util.cv_show(img, "img")

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
util.cv_show(sobelx, 'sobelx')

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
util.cv_show(sobely, 'sobely')
```
原图：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-33-40-1650422020414-image-20220420103340305.png" alt="image-20220420103340305" style="zoom:35%;" />
执行Sobel运算后：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-34-07-1650422047978-image-20220420103407876.png" alt="image-20220420103407876" style="zoom:35%;" />
<img src="http://img.yibogame.com/uPic/2022-04-20-10-41-05-1650422465326-image-20220420104105203.png" alt="image-20220420104105203" style="zoom:35%;" />

负值：如果x方向的“左边 - 右边”后为负数或y方向的“下面 - 上面”后为负数，负数无法显示。所以，将方法名改为绝对值方式`cv2.convertScaleAbs(src, dst=None, alpha=None, beta=None)`
```python
sobely = cv2.convertScaleAbs(sobely)
util.cv_show(sobely, 'sobely')
```
执行后`sobely`结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-48-23-1650422903629-image-20220420104823524.png" alt="image-20220420104823524" style="zoom:35%;" />

* 分别计算x和y：
```py
img = cv2.imread('res/pie.png', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
util.cv_show(sobelxy, 'sobelxy')
```
执行后结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-10-58-47-1650423527516-image-20220420105847411.png" alt="image-20220420105847411" style="zoom:35%;" />
* 直接计算（不推荐。从效果上来说要比分别计算要差一点）：
```python
img = cv2.imread('res/pie.png', cv2.IMREAD_GRAYSCALE)
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
util.cv_show(sobelxy, 'sobelxy')
```
执行后结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-11-00-07-1650423607372-image-20220420110007266.png" alt="image-20220420110007266" style="zoom:35%;" />

#### 图像梯度----Scharr算子

公式：
<img src="http://img.yibogame.com/uPic/2022-04-20-11-03-21-1650423801990-scharr.png" alt="scharr" style="zoom:55%;" />
与Sobel算子对比的话，数值相对更大，算法一样。



> 图像梯度----laplacian算子
> 公式：
> <img src="http://img.yibogame.com/uPic/2022-04-20-11-05-26-1650423926909-l.png" alt="l" style="zoom:70%;" />
> G = (p2 + p4 + p6  +p8) - 4 * p5

三种算子的对比：
```python
# 原图
img = cv2.imread('res/lena.jpg', cv2.IMREAD_GRAYSCALE)
util.cv_show(img, 'img')
# 不同算子的差异
img = cv2.imread('res/lena.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
util.cv_show(res, 'res')
```
原图：
<img src="http://img.yibogame.com/uPic/2022-04-20-15-39-00-1650440340511-image-20220420153900405.png" alt="image-20220420153900405" style="zoom:35%;" />

各算子执行运算后结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-15-36-58-1650440218076-image-20220420153657956.png" alt="image-20220420153657956" style="zoom:35%;" />

#### Canny边缘检测
1. 使用高斯滤波器，以平滑图像，滤除噪声。
2. 计算图像中每个像素点的梯度强度和方向。
3. 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。（选最符合的，删除其他）
4. 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。（再次过滤）
5. 通过抑制孤立的弱边缘最终完成边缘检测。

   

* 高斯滤波器
<img src="http://img.yibogame.com/uPic/2022-04-20-15-55-10-1650441310588-canny_1.png" alt="canny_1" style="zoom:50%;" />

* 梯度和方向
<img src="http://img.yibogame.com/uPic/2022-04-20-15-55-35-1650441335087-canny_2.png" alt="canny_2" style="zoom:50%;" />

* 非极大值抑制
	1. 方法1
	<img src="http://img.yibogame.com/uPic/2022-04-20-16-09-20-1650442160179-canny_3.png" alt="canny_3" style="zoom:67%;" />
	2. 方法2
	<img src="http://img.yibogame.com/uPic/2022-04-20-16-10-45-1650442245435-canny_4.png" alt="canny_4" style="zoom: 50%;" />

* 双阈值检测
![canny_5](http://img.yibogame.com/uPic/2022-04-20-16-25-54-1650443154196-canny_5.png)

```python
img = cv2.imread("res/lena.jpg", cv2.IMREAD_GRAYSCALE)

v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((img, v1, v2))
util.cv_show(res, 'res')

# %%
img = cv2.imread("res/car.png", cv2.IMREAD_GRAYSCALE)

v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((img, v1, v2))
util.cv_show(res, 'res')
```
执行后结果：
<img src="http://img.yibogame.com/uPic/2022-04-20-16-29-06-1650443346251-image-20220420162906135.png" alt="image-20220420162906135" style="zoom:50%;" />

<img src="http://img.yibogame.com/uPic/2022-04-20-16-29-26-1650443366902-image-20220420162926780.png" alt="image-20220420162926780" style="zoom:50%;" />

#### 图像金字塔
* 高斯金字塔
	<img src="http://img.yibogame.com/uPic/2022-04-20-16-38-40-1650443920528-Pyramid_1.png" alt="Pyramid_1" style="zoom:50%;" />
	* 向下采样方法（缩小，越采样大小越小）
	<img src="http://img.yibogame.com/uPic/2022-04-20-16-40-13-1650444013735-Pyramid_2.png" alt="Pyramid_2" style="zoom:50%;" />
	
	* 向上采样方法（放大，越采样大小越大）
	<img src="http://img.yibogame.com/uPic/2022-04-20-16-40-37-1650444037202-Pyramid_3.png" alt="Pyramid_3" style="zoom:50%;" />
	
	```python
  img = cv2.imread("res/AM.png")
  util.cv_show(img, 'img')
  print(img.shape)
	
  up = cv2.pyrUp(img)
  util.cv_show(up, 'up')
  print(up.shape)
	
	down = cv2.pyrDown(img)
	util.cv_show(down, 'down')
	print(down.shape)
	```
	
	输出：

	```python
	(442, 340, 3)
	(884, 680, 3)
	(221, 170, 3)
	```
	
* 拉普拉斯金字塔
	<img src="http://img.yibogame.com/uPic/2022-04-20-16-55-05-1650444905788-Pyramid_4.png" alt="Pyramid_4" style="zoom:50%;" />
	G0、Gi：原始输入图像。
	
	```python
	down=cv2.pyrDown(img)
  down_up=cv2.pyrUp(down)
  l_1=img-down_up
  cv_show(l_1,'l_1')
	```


#### 图像轮廓
###### `cv2.findContours(img,mode,method)` 

注：轮廓包含**内轮廓**与**外轮廓**，除非你指定只检索外轮廓。

* `mode`:轮廓检索模式
	* `RETR_EXTERNAL` ：只检索最外面的轮廓；
	* `RETR_LIST`：检索所有的轮廓，并将其保存到一条链表当中；
	* `RETR_CCOMP`：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
	* `RETR_TREE`：检索所有的轮廓，并重构嵌套轮廓的整个层次;

* `method`:轮廓逼近方法
	* `CHAIN_APPROX_NONE`：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
	* `CHAIN_APPROX_SIMPLE`:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。

	1. 二值化处理：
  ```python
  img = cv2.imread('res/contours.png')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  util.cv_show(thresh, 'thresh')
  ```

  <img src="http://img.yibogame.com/uPic/2022-04-21-12-06-59-1650514019853-image-20220421120659726.png" alt="image-20220421120659726" style="zoom:30%;" />
  2. 找到轮廓
  ```python
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  ```
  3. 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
  ```python
    #注意需要copy,要不原图会变。。。
    draw_img = img.copy()
    #参数-1表示画所有轮廓
    res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
    util.cv_show(res, 'res')
  ```
  <img src="http://img.yibogame.com/uPic/2022-04-21-12-10-57-1650514257367-image-20220421121057257.png" alt="image-20220421121057257" style="zoom: 30%;" />
  4. 只画指定的某个轮廓
  ```python
  draw_img = img.copy()
  #第一个参数0表示第0个轮廓
  res = cv2.drawContours(draw_img, contours, 0, (0, 0, 255), 2)
  util.cv_show(res, 'res')
  ```
  <img src="http://img.yibogame.com/uPic/2022-04-21-12-13-01-1650514381412-image-20220421121301300.png" alt="image-20220421121301300" style="zoom:30%;" />
  

###### 轮廓特征计算
```python
	cnt = contours[0]
	# 面积
	area = cv2.contourArea(cnt)
	print("area", area)  # 输出：area 8500.5
	# 周长，True表示闭合的
	perimeter = cv2.arcLength(cnt, True)
	print("perimeter", perimeter)   # 输出：perimeter 437.9482651948929
```

#### 轮廓近似

<img src="http://img.yibogame.com/uPic/2022-04-21-15-33-00-1650526380906-2022-04-21-13-45-16-1650519916328-image-20220421134516228.png" alt="2022-04-21-13-45-16-1650519916328-image-20220421134516228" style="zoom:30%;" />

```python
img = cv2.imread('res/contours2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
util.cv_show(res, 'res')
###########
epsilon = 0.15 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)	#近似

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
util.cv_show(res, 'res')
```

<img src="http://img.yibogame.com/uPic/2022-04-21-13-17-37-1650518257552-image-20220421131737425.png" alt="image-20220421131737425" style="zoom:33%;" />

<img src="http://img.yibogame.com/uPic/2022-04-21-13-17-50-1650518270602-image-20220421131750478.png" alt="image-20220421131750478" style="zoom:33%;" />

#### 边界矩形
  ```python
  img = cv2.imread('res/contours.png')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  cnt = contours[0]
	# 外接矩形
  x, y, w, h = cv2.boundingRect(cnt)
  img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
  util.cv_show(img, 'img')

  area = cv2.contourArea(cnt)
  x, y, w, h = cv2.boundingRect(cnt)
  rect_area = w * h
  extent = float(area) / rect_area
  print('轮廓面积与边界矩形比', extent)	#输出：轮廓面积与边界矩形比 0.5154317244724715

  # 外接圆
  (x, y), radius = cv2.minEnclosingCircle(cnt)
  center = (int(x), int(y))
  radius = int(radius)
  img = cv2.circle(img, center, radius, (0, 255, 0), 2)
  util.cv_show(img, 'img')
  ```

<img src="http://img.yibogame.com/uPic/2022-04-21-15-35-58-1650526558551-image-20220421153558433.png" alt="image-20220421153558433" style="zoom:30%;" />

<img src="http://img.yibogame.com/uPic/2022-04-21-15-36-14-1650526574717-image-20220421153614590.png" alt="image-20220421153614590" style="zoom:30%;" />

#### 模板匹配
模板匹配和卷积原理很像，模板在原图像上从原点开始滑动，计算模板与（图像被模板覆盖的地方）的差别程度，这个差别程度的计算方法在opencv里有6种，然后将每次计算的结果放入一个矩阵里，作为结果输出。假如原图形是AxB大小，而模板是axb大小，则输出结果的矩阵是(A-a+1)x(B-b+1)
* `TM_SQDIFF`：计算平方不同，计算出来的值越小，越相关
* `TM_CCORR`：计算相关性，计算出来的值越大，越相关
* `TM_CCOEFF`：计算相关系数，计算出来的值越大，越相关
* `TM_SQDIFF_NORMED`：计算归一化平方不同，计算出来的值越接近0，越相关
* `TM_CCORR_NORMED`：计算归一化相关性，计算出来的值越接近1，越相关
* `TM_CCOEFF_NORMED`：计算归一化相关系数，计算出来的值越接近1，越相关

[公式]:https://docs.opencv.org/3.3.1/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d	"官方文档3.3.1"

```python
# 模板匹配
img = cv2.imread('res/lena.jpg', 0)
template = cv2.imread('res/face.jpg', 0)
h, w = template.shape[:2]
print("h=", h, "w=", w)
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
print("res.shape:", res.shape)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print("minMaxLoc:", min_val, max_val, min_loc, max_loc)
# 输出：
# h= 110 w= 85
# res.shape: (154, 179)
# minMaxLoc: 39168.0 74403584.0 (107, 89) (159, 62)
```
通过各种方法的计算：
```python
img = cv2.imread('res/lena.jpg', 0)
template = cv2.imread('res/face.jpg', 0)
h, w = template.shape[:2]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
img2 = img.copy()

	# 匹配方法的真值
method = eval(meth)
print(method)
res = cv2.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	# 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
top_left = min_loc
else:
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

	# 画矩形
cv2.rectangle(img2, top_left, bottom_right, 255, 2)

plt.subplot(121), plt.imshow(res, cmap='gray')
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.subplot(122), plt.imshow(img2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.suptitle(meth)
plt.show()
```
<img src="http://img.yibogame.com/uPic/2022-04-21-16-13-57-1650528837135-image-20220421161357011.png" alt="image-20220421161357011" style="zoom:30%;" />
<img src="http://img.yibogame.com/uPic/2022-04-21-16-14-17-1650528857659-image-20220421161417546.png" alt="image-20220421161417546" style="zoom:30%;" />
<img src="http://img.yibogame.com/uPic/2022-04-21-16-14-39-1650528879947-image-20220421161439831.png" alt="image-20220421161439831" style="zoom:30%;" />
<img src="http://img.yibogame.com/uPic/2022-04-21-16-14-53-1650528893291-image-20220421161453170.png" alt="image-20220421161453170" style="zoom:30%;" />
<img src="http://img.yibogame.com/uPic/2022-04-21-16-15-08-1650528908393-image-20220421161508276.png" alt="image-20220421161508276" style="zoom:30%;" />
<img src="http://img.yibogame.com/uPic/2022-04-21-16-15-21-1650528921974-image-20220421161521859.png" alt="image-20220421161521859" style="zoom:30%;" />


#### 匹配多个对象
```python
img_rgb = cv2.imread('res/mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('res/mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
	# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
bottom_right = (pt[0] + w, pt[1] + h)
cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)
```

<img src="http://img.yibogame.com/uPic/2022-04-21-16-35-35-1650530135934-image-20220421163535816.png" alt="image-20220421163535816" style="zoom:50%;" />

#### 直方图

<img src="http://img.yibogame.com/uPic/2022-04-21-20-53-33-1650545613862-hist_1.png" alt="hist_1" style="zoom:50%;" />

按0~255（可通过`histSize`改变组数）从左到右的统计出现的次数。
#### `cv2.calcHist(images,channels,mask,histSize,ranges)`
`images`: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应该用中括号 [] 括起来例如[img]
`channels`: 同样用中括号括起来它会告函数我们统幅图 像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0]、[1]、[2] 它们分别对应着 BGR。
`mask`: 掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
`histSize`:BIN 的数目。也应用中括号括起来
`ranges`: 像素值范围常为 [0~256]
```python
img = cv2.imread('res/cat.jpg', 0)  # 0表示灰度图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])# 参数[0]:因为是灰度图，所以通道有且只有0
print("hist.shape:", hist.shape) # 输出:hist.shape: (256, 1)
plt.hist(img.ravel(), 256)
plt.show()
```

<img src="http://img.yibogame.com/uPic/2022-04-21-21-05-22-1650546322852-image-20220421210522734.png" alt="image-20220421210522734" style="zoom:35%;" />
```python
img = cv2.imread('res/cat.jpg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
histr = cv2.calcHist([img], [i], None, [256], [0, 256])
plt.plot(histr, color=col)
plt.xlim([0, 256])
plt.show()
```

<img src="http://img.yibogame.com/uPic/2022-04-21-21-10-47-1650546647570-image-20220421211047455.png" alt="image-20220421211047455" style="zoom:35%;" />

###### Mask操作

```python
img = cv2.imread('res/cat.jpg')

	# 创建mask
mask = np.zeros(img.shape[:2], np.uint8) # unit8存放的大小是0~255
print(mask.shape) # 输出：(414, 500)
mask[100:300, 100:400] = 255 # 把需要保存的地方的值设置为白色，也就是255

	# 应用mask后的img
masked_img = cv2.bitwise_and(img, img, mask=mask)  # 与操作
	# 不带mask的直方图
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
	# 带mask的直方图
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()
```
执行结果：
<img src="http://img.yibogame.com/uPic/2022-04-21-21-16-25-1650546985689-image-20220421211625558.png" alt="image-20220421211625558" style="zoom:45%;" />

#### 直方图均衡化

<img src="http://img.yibogame.com/uPic/2022-04-21-21-28-21-1650547701425-hist_2.png" alt="hist_2" style="zoom:80%;" />

<img src="http://img.yibogame.com/uPic/2022-04-21-21-28-42-1650547722989-hist_3.png" alt="hist_3" style="zoom:80%;" />

<img src="http://img.yibogame.com/uPic/2022-04-21-21-28-53-1650547733064-hist_4.png" alt="hist_4" style="zoom:80%;" />

```py
img = cv2.imread('res/clahe.jpg', 0)  # 0表示灰度图 #clahe
plt.hist(img.ravel(), 256)
plt.show()

equ = cv2.equalizeHist(img)	# 直方图均衡化
plt.hist(equ.ravel(), 256)
plt.show()

res = np.hstack((img, equ))
util.cv_show(res, 'res')
```

运行后结果：
<img src="http://img.yibogame.com/uPic/2022-04-21-21-38-03-1650548283452-image-20220421213803331.png" alt="image-20220421213803331" style="zoom:35%;" />
<img src="http://img.yibogame.com/uPic/2022-04-21-21-38-23-1650548303899-image-20220421213823799.png" alt="image-20220421213823799" style="zoom:35%;" />
<img src="http://img.yibogame.com/uPic/2022-04-21-21-38-42-1650548322047-image-20220421213841917.png" alt="image-20220421213841917" style="zoom:35%;" />

#### 自适应直方图均衡化

```py
img = cv2.imread('res/clahe.jpg', 0)  # 0表示灰度图 #clahe

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 自适应均衡化，根据创建的小格子分别做均衡化(tileGridSize参数)
res_clahe = clahe.apply(img)

equ = cv2.equalizeHist(img)
res = np.hstack((img, equ, res_clahe))
util.cv_show(res, 'res')
```

<img src="http://img.yibogame.com/uPic/2022-04-21-21-45-48-1650548748027-image-20220421214547876.png" alt="image-20220421214547876" style="zoom:50%;" />

#### 傅里叶变换

傅里叶变换是什么？
<img src="http://img.yibogame.com/uPic/2022-04-21-21-53-33-1650549213356-4695ce06197677bab880cd55b6846f12_1440w.jpg" alt="img" style="zoom: 67%;" />

[知乎]: https://zhuanlan.zhihu.com/p/19763358	"傅里叶分析之掐死教程（完整版）更新于2014.06.06"

傅里叶变换有什么作用？
* 高频：变化剧烈的灰度分量，例如边界
* 低频：变化缓慢的灰度分量，例如一片大海

#### 滤波
* 低通滤波器：只保留低频，会使得图像模糊
* 高通滤波器：只保留高频，会使得图像细节增强

opencv中主要就是cv2.dft()和cv2.idft()，输入图像需要先转换成np.float32 格式。
得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）。
```python
img = cv2.imread('res/lena.jpg', 0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft) # 将低频的值从左上角转换到中心位置
	# 得到灰度图能表示的形式
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

<img src="http://img.yibogame.com/uPic/2022-04-21-22-01-20-1650549680927-image-20220421220120808.png" alt="image-20220421220120808" style="zoom:45%;" />

```python
img = cv2.imread('res/lena.jpg', 0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)	# 从左上角变换移动到中间

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

	# 低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)	# 0是需要保留的，其余的地方仍掉
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
	# 高通滤波
  # mask = np.ones((rows, cols, 2), np.uint8)
	# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

	# IDFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)	# 还原到左上角
img_back = cv2.idft(f_ishift)	# 再做逆变换处理为图像（这个图像仅仅是实部与虚部的，无法直接看）
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])	# 把实部与虚部的图像转为普通图片

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()
```
低通滤波后结果：
<img src="http://img.yibogame.com/uPic/2022-04-21-22-07-57-1650550077628-image-20220421220757507.png" alt="image-20220421220757507" style="zoom:50%;" />
高通滤波后结果：
<img src="http://img.yibogame.com/uPic/2022-04-21-22-16-00-1650550560435-image-20220421221600314.png" alt="image-20220421221600314" style="zoom:50%;" />
