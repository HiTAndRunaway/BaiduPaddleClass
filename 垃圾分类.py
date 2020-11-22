#!/usr/bin/env python
# coding: utf-8

# ## 试题说明
# 
# ### 任务描述
# 近年来，随着人工智能的发展，其在语音识别、自然语言处理、图像与视频分析等诸多领域取得了巨大成功。随着政府对环境保护的呼吁，垃圾分类成为一个亟待解决的问题，本次竞赛将聚焦在垃圾图片的分类，利用人工智能技术，对居民生活垃圾图片进行检测，找出图片中有哪些类别的垃圾。
# 要求参赛者给出一个算法或模型，对于给定的图片，检测出图片中的垃圾类别。给定图片数据，选手据此训练模型，为每张测试数据预测出最正确的类别。
# 
# ### 数据说明
# 本竞赛所用训练和测试图片均来自生活场景。总共四十个类别，类别和标签对应关系在训练集中的dict文件里。图片中垃圾的类别，格式是“一级类别/二级类别”，二级类别是具体的垃圾物体类别，也就是训练数据中标注的类别，比如一次性快餐盒、果皮果肉、旧衣服等。一级类别有四种类别：可回收物、厨余垃圾、有害垃圾和其他垃圾。
# 
# 数据文件包括训练集(有标注)和测试集(无标注)，训练集的所有图片分别保存在train文件夹下面的0-39个文件夹中，文件名即类别标签，测试集共有400张待分类的垃圾图片在test文件夹下，testpath.txt保存了所有测试集文件的名称，格式为：name+\n。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/cec625e7b61d459fa13ed2b822817bfd57030ad23c524f1ab93b3e7108d75d78)
# ![](https://ai-studio-static-online.cdn.bcebos.com/df06e2956f2044fab21fde2a9602abfdb9d4720861d54c66b2e079f3e64000a7)
# 
# ### 提交答案
# 考试提交，需要提交**模型代码项目版本**和**结果文件**。结果文件为TXT文件格式，命名为model_result.txt，文件内的字段需要按照指定格式写入。
# 
# 提交结果的格式如下：
# 1. 每个类别的行数和测试集原始数据行数应一一对应，不可乱序。
# 2. 输出结果应检查是否为400行数据，否则成绩无效。
# 3. 输出结果文件命名为model_result.txt，一行一个类别标签（数字）
# 
# 
# 样例如下：
# 
# ···
# 
# 35
# 
# 3
# 
# 2
# 
# 37
# 
# 10
# 
# 3
# 
# 26
# 
# 4
# 
# 34
# 
# 21
# 
# ···
# 

# ### 开始答题
# #Step1、基础工作
# #加载数据文件
# 
# 导入python包

# In[1]:


# 导入需要的包
import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import matplotlib.pyplot as plt

'''
参数配置
'''
batch_size = 96
num_epochs = 20
train_parameters = {
    "input_size": [3, 224, 224],  # 输入图片的shape
    "class_dim": -1,  # 分类数
    "src_path": "/home/aistudio/data/data35095/train.zip",  # 原始数据集路径
    "target_path": "/home/aistudio/data/garbage/",  # 要解压的路径
    "train_list_path": "./train_data.txt",  # train_data.txt路径
    "eval_list_path": "./eval_data.txt",  # eval_data.txt路径
    "readme_path": "/home/aistudio/data/readme.json",  # readme.json路径
    "label_dict": {},  # 标签字典
    "image_count": -1,  # 训练图片数量
    "train_batch_size": batch_size,  # 训练时每个批次的大小
    "num_epochs": num_epochs,  # 训练轮数
    "mode": "train",  # 工作模式

    "network_resnet": {  # ResNet
        "layer": 50  # ResNet的层数
    },
    "continue_train": False,  # 是否接着上一次保存的参数接着训练
    "regenerat_imgs": False,  # 是否生成增强图像文件，True强制重新生成，慢
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值
    "use_gpu": True,
    "use_image_enhance": True,
    "image_enhance_strategy": {  # 图像增强相关策略
        "need_distort": True,  # 是否启用图像颜色增强
        "need_rotate": True,  # 是否需要增加随机角度
        "need_crop": True,  # 是否要增加裁剪
        "need_flip": True,  # 是否要增加水平随机翻转
        "need_expand": True,  # 是否要增加扩展
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.000125,  # 超参数学习率
        "name": "cosine_decay",
        "batch_size": batch_size,
        "epochs": [40, 80, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "good_acc1": 0.9975  # 0.92
    },
    "rms_strategy": {
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "momentum_strategy": {
        # "learning_rate": 0.001,
        "learning_rate": 0.0001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "sgd_strategy": {
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "adam_strategy": {
        "learning_rate": 0.002
    },
    "adamax_strategy": {
        "learning_rate": 0.00125
    }
}


# In[2]:


def unzip_data(src_path, target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至data目录下
    '''
    if not os.path.isdir(target_path):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


src_path = train_parameters["src_path"]
target_path = train_parameters["target_path"]
unzip_data(src_path, target_path)

# 数据增强

# In[3]:


get_ipython().system('pip install imgaug')


# In[4]:


def distort_image(img):
    """
    图像增强
    :param img:
    :return:
    """

    def random_brightness(img):
        """
        随机亮度调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
            brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']
            delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
            img = ImageEnhance.Brightness(img).enhance(delta)
        return img

    def random_contrast(img):
        """
        随机对比度调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
            contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']
            delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
            img = ImageEnhance.Contrast(img).enhance(delta)
        return img

    def random_saturation(img):
        """
        随机饱和度调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
            saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']
            delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
            img = ImageEnhance.Color(img).enhance(delta)
        return img

    def random_hue(img):
        """
        随机色调整
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_enhance_strategy']['hue_prob']:
            hue_delta = train_parameters['image_enhance_strategy']['hue_delta']
            delta = np.random.uniform(-hue_delta, hue_delta)
            img_hsv = np.array(img.convert('HSV'))
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
            img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
        return img

    ops = [random_brightness, random_contrast, random_saturation, random_hue]
    np.random.shuffle(ops)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = ops[3](img)
    return img


def random_crop(img, scales=[0.3, 1.0], max_ratio=2.0, max_trial=50):
    """
    随机裁剪
    :param img:
    :param scales:
    :param max_ratio:
    :param constraints:
    :param max_trial:
    :return:
    """
    if random.random() > 0.6:
        return img

    w, h = img.size
    crops = [(0, 0, w, h)]

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        return img
    return img


def random_expand(img, keep_ratio=True):
    """
    随机扩张
    :param img:
    :param keep_ratio:
    :return:
    """
    if np.random.uniform(0, 1) < train_parameters['image_enhance_strategy']['expand_prob']:
        return img

    max_ratio = train_parameters['image_enhance_strategy']['expand_max_ratio']
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_parameters['mean_rgb'][i]

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img

    return Image.fromarray(out_img)


def random_flip(img, thresh=0.5):
    """
    随机翻转
    :param img:
    :param thresh:
    :return:
    """
    if random.random() > thresh:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


def rotate_image(img, thresh=0.5):
    """
    图像增强，增加随机旋转角度
    """
    if random.random() > thresh:
        angle = np.random.randint(-14, 15)
        img = img.rotate(angle)

    return img


def preprocess(img, mode='train'):
    """
    preprocess，图像增强预处理
    :param img:
    :param mode:
    :return:
    """
    # 在线增强模式和离线增强模式可以叠加使用，但是个人不建议一开始就用，样本文件数量多一些的话，耗时太久了。
    # mode == 'train'，训练模式下才进行在线图像增强    
    if mode == 'train':
        if train_parameters["image_enhance_strategy"]["need_distort"]:
            img = distort_image(img)
        if train_parameters["image_enhance_strategy"]['need_rotate']:
            img = rotate_image(img, thresh=0.5)
        if train_parameters["image_enhance_strategy"]["need_crop"]:
            img = random_crop(img)
        if train_parameters["image_enhance_strategy"]["need_flip"]:
            img = random_flip(img)
        if train_parameters["image_enhance_strategy"]["need_expand"]:
            img = random_expand(img)
    return img


# 准备txt:训练集、验证集

# In[5]:


def get_data_list(target_path, train_list_path, eval_list_path):
    '''
    生成数据列表
    '''
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path + 'train/'
    class_dirs = os.listdir(data_list_path)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    # class_label=0
    # 存放类别数目
    class_dim = 0
    # 存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":

            class_dim += 1
            # 每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_list_path + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                name_path = path + '/' + img_path  # 每张图片的路径
                if class_sum % 10 == 0:  # 每8张图片取一个做验证数据
                    eval_sum += 1  # test_sum为测试数据的数目
                    eval_list.append(name_path + "\t" + class_dir + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t" + class_dir + "\n")  # trainer_sum测试数据的数目

                    """
                    #离线增强方式：使用图像增强，增加样本文件数，注意只能在训练集文件列表中进行增强
                    #这里只生成了一个增强后的文件，并添加到训练集文件列表中，
                    #如果需要更多的增强文件，可以复制代码执行或者做一个循环处理（preprocess函数内部有随机处理机制，每次调用生成的图像不同）
                    if (train_parameters["use_image_enhance"]):
                        img = Image.open(name_path) 
                        if img.mode != 'RGB': 
                            img = img.convert('RGB')                        
                        #强制生成标识或者没有预生成文件
                        if train_parameters["regenerat_imgs"] or not os.path.exists(name_path2):
                            img = preprocess(img)    #'train'模式
                            img.save(name_path2)   
                        trainer_sum += 1 
                        trainer_list.append(name_path2+ "\t%d" % class_label + "\n") #trainer_sum测试数据的数目
                        #
                        all_class_images += 1                                   #所有类图片的数目
                    # 
                    """
                class_sum += 1  # 每类图片的数目
                all_class_images += 1  # 所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir  # 类别名称
            class_detail_list['class_label'] = train_parameters['label_dict'][class_dir]  # 类别标签
            class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
            class_detail.append(class_detail_list)
            # 初始化标签列表
            # train_parameters['label_dict'][str(class_label)] = class_dir
            # class_label += 1

    # 初始化分类数
    train_parameters['class_dim'] = class_dim

    # 乱序
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

            # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path  # 文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)

    print('生成数据列表完成！')


# In[6]:


import random

'''
参数初始化
'''
src_path = train_parameters['src_path']
target_path = train_parameters['target_path']
train_list_path = target_path + train_parameters['train_list_path']
eval_list_path = target_path + train_parameters['eval_list_path']
batch_size = train_parameters['train_batch_size']

with open(train_parameters['target_path'] + 'garbage_dict.json') as f:
    train_parameters['label_dict'] = json.load(f)

with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
get_data_list(target_path, train_list_path, eval_list_path)


# Step2、定义reader

# In[7]:


# reader
def custom_reader(file_list, mode):
    '''
    自定义data_reader
    '''

    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            ## 打乱次序
            random.shuffle(lines)
            for line in lines:
                if mode == 'train' or mode == 'eval':
                    img_path, lab = line.strip().split('\t')
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                        # 使用在线增强方式，不用的话，把下一段代码注释掉
                    # 验证模式下不用增强，所以加了一个工作模式条件判断
                    """                    
                    if mode == 'train': 
                        #在线增强方式：
                        if (train_parameters["use_image_enhance"]):
                            img = preprocess(img, mode)  #只有在'train'模式下才执行图像增强
                    """
                    # 图像缩放到指定大小，VGG是3x224x224
                    img = img.resize((224, 224), Image.ANTIALIAS)  ##BILINEAR
                    img = np.array(img).astype('float32')
                    # 图像数据按照所需要的格式重新排列
                    img = img.transpose((2, 0, 1))  # HWC to CHW 
                    img = img / 255.0  # 像素值归一化
                    yield img, int(lab)
                elif mode == 'test':
                    img_path = line.strip()
                    img = Image.open(img_path)
                    img = img.resize((224, 224), Image.ANTIALIAS)
                    img = np.array(img).astype('float32')
                    img = img.transpose((2, 0, 1))  # HWC to CHW 
                    img = img / 255.0  # 像素值归一化
                    yield img

    return reader


# In[8]:


'''
构造数据提供器
'''
# 训练集和验证集调用同样的函数，但是工作模式这个参数不一样。
train_reader = paddle.batch(custom_reader(train_list_path, 'train'),
                            batch_size=batch_size,
                            drop_last=True)
eval_reader = paddle.batch(custom_reader(eval_list_path, 'eval'),
                           batch_size=batch_size,
                           drop_last=True)

# In[9]:


'''
PLOT画图
'''
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []


def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()


def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()


# Step2、定义模型

# In[10]:


# ResNet网络定义
import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework
import math
import sys
from paddle.fluid.param_attr import ParamAttr


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                self.full_name(),
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=25):
        super(ResNet, self).__init__(name_scope)

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, "supported layers are {} but input layer is {}".format(supported_layers,
                                                                                                  layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            self.full_name(),
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(input_dim=num_channels,
                          output_dim=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs, label=None):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(x=y, shape=[-1, y.shape[1]])
        y = self.out(y)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y


# In[11]:


# 看一下网络参数
with fluid.dygraph.guard():
    network = ResNet('resnet', layers=50, class_dim=40)
    img = np.zeros([1, 3, 224, 224]).astype('float32')
    img = fluid.dygraph.to_variable(img)
    outs = network(img).numpy()
    print(outs)

# In[12]:


# 定义eval_net函数
print(train_parameters)


def eval_net(reader, model):
    acc_set = []

    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int')
        y_data = y_data[:, np.newaxis]
        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)

        #        out, acc = model(img, label)
        #        lab = np.argsort(out.numpy())
        #        accs.append(acc.numpy()[0]) 

        acc_set.append(float(acc.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()

    return acc_val_mean


# step3、训练模型

# In[13]:


'''
模型训练，继续炼丹
'''
# train_parameters["num_epochs"] =20 #临时改了一个数值
epochs_num = train_parameters["num_epochs"]
batch_size = train_parameters["train_batch_size"]  # train_parameters["learning_strategy"]["batch_size"]
total_images = train_parameters["image_count"]
stepsnumb = int(math.ceil(float(total_images) / batch_size))

# resnet层数定义，要改一下
# train_parameters["layer"] = 50

with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):  # 使用GPU进行训练
    # with fluid.dygraph.guard():                            #使用CPU进行训练
    print('class_dims:', train_parameters['class_dim'])
    print('label_dict:', train_parameters['label_dict'])

    best_acc = 0
    best_epc = -1
    eval_epchnumber = 0
    all_eval_avgacc = []
    all_eval_iters = []

    all_train_iter = 0
    all_train_iters = []
    all_train_costs = []
    all_train_accs = []

    model = ResNet("resnet", train_parameters["network_resnet"]["layer"], train_parameters["class_dim"])

    # """
    if True:
        try:
            if os.path.exists('MyResNet_best.pdparams'):
                print('try model file MyResNet_best. Loading...')
                model_dict, _ = fluid.load_dygraph('MyResNet_best')
                if os.path.exists('beast_acc_ResNet.txt'):
                    with open('beast_acc_ResNet.txt', "r") as f:
                        best_acc = float(f.readline())
            else:
                print('try model file MyResNet. Loading...')
                model_dict, _ = fluid.load_dygraph('MyResNet')
                if os.path.exists('acc_ResNet.txt'):
                    with open('acc_ResNet.txt', "r") as f:
                        best_acc = float(f.readline())
            # 防止上一次acc太大，导致本次训练结果不存储了
            start_acc = min(0.92, train_parameters["early_stop"]["good_acc1"])
            if best_acc >= start_acc:
                best_acc = start_acc
            model.load_dict(model_dict)  # 加载模型参数
        except Exception as e:
            print(e)
        print('model initialization finished.')
    # """

    # 后面代码会切换工作模式
    model.train()  # 训练模式

    # 定义优化方法optimizer_momentum_setting, optimizer_sgd_setting, optimizer_rms_setting, optimizer_adam_setting, optimizer_Adamax_setting
    # optimizer = optimizer_momentum_setting(model.parameters())

    paramsList = model.parameters()
    params = train_parameters
    total_images = params["image_count"]
    ls = params["learning_strategy"]
    batch_size = ls["batch_size"]
    step = int(math.ceil(float(total_images) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    lr = 0.0002  # params["learning_strategy"]["lr"]  #0.00125
    num_epochs = params["num_epochs"]
    regularization = fluid.regularizer.L2Decay(regularization_coeff=0.1)
    learning_rate = lr
    ##learning_rate=fluid.layers.cosine_decay(
    ##    learning_rate=lr, step_each_epoch=step, epochs=num_epochs)
    # momentum_rate = 0.9
    # optimizer = fluid.optimizer.Momentum(learning_rate=learning_rate,momentum=momentum_rate,regularization=regularization,parameter_list=paramsList)
    # optimizer=fluid.optimizer.SGDOptimizer(learning_rate=learning_rate, regularization=regularization, parameter_list=paramsList)
    optimizer = fluid.optimizer.AdamaxOptimizer(learning_rate=learning_rate, regularization=regularization,
                                                parameter_list=paramsList)
    # optimizer=fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, regularization=regularization, parameter_list=paramsList)

    # epochs_num = 1
    # 开始训练
    for epoch_num in range(epochs_num):
        model.train()  # 训练模式
        # 从train_reader中获取每个批次的数据
        for batch_id, data in enumerate(train_reader()):
            # dy_x_data = np.array([x[0] for x in data]).astype('float32')
            # y_data = np.array([x[1] for x in data]).astype('int')
            dy_x_data = np.array([x[0] for x in data]).astype('float32').reshape(-1, 3, 224, 224)
            y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

            # 将Numpy转换为DyGraph接收的输入
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)

            out, acc = model(img, label)
            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)

            # 使用backward()方法可以执行反向网络
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            # 将参数梯度清零以保证下一轮训练的正确性
            model.clear_gradients()

            all_train_iter = all_train_iter + train_parameters['train_batch_size']
            all_train_iters.append(all_train_iter)
            all_train_costs.append(loss.numpy()[0])
            all_train_accs.append(acc.numpy()[0])

            dy_param_value = {}
            for param in model.parameters():
                dy_param_value[param.name] = param.numpy

            if batch_id % 100 == 0 or batch_id == stepsnumb - 1:
                print("epoch %3d step %4d: loss: %f, acc: %f" % (epoch_num, batch_id, avg_loss.numpy(), acc.numpy()))

        if epoch_num % 1 == 0 or epoch_num == epochs_num - 1:
            model.eval()
            epoch_acc = eval_net(eval_reader, model)
            print('  train_pass:%d,eval_acc=%f' % (epoch_num, epoch_acc))
            eval_epchnumber = epoch_num
            all_eval_avgacc.append(epoch_acc)
            all_eval_iters.append([eval_epchnumber, epoch_acc])

            if best_acc < epoch_acc:
                best_epc = epoch_num
                best_acc = epoch_acc
                # 保存模型参数，对应当前最好的评估结果
                fluid.save_dygraph(model.state_dict(), 'MyResNet_best')
                print('    current best_eval_acc=%f in No.%d epoch' % (best_acc, best_epc))
                print('    MyResNet_best模型已保存')
                with open('beast_acc_ResNet.txt', "w") as f:
                    f.write(str(best_acc))
                # fluid.dygraph.save_dygraph(model.state_dict(), "save_dir/ResNet/model_best")
                # fluid.dygraph.save_dygraph(optimizer.state_dict(), "save_dir/ResNet/model_best")

            # 训练过程结果显示
            # draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning loss","trainning acc")

    # 保存模型参数，但不一定是最好评估结果对应的模型
    fluid.save_dygraph(model.state_dict(), "MyResNet")
    print('MyResNetG模型已保存')
    print("Final loss: {}".format(avg_loss.numpy()))
    # fluid.dygraph.save_dygraph(model.state_dict(), "save_dir/model")
    # fluid.dygraph.save_dygraph(optimizer.state_dict(), "save_dir/model")
    with open('acc_ResNet.txt', "w") as f:
        f.write(str(epoch_acc))

    draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning loss", "trainning acc")
    draw_process("trainning loss", "red", all_train_iters, all_train_costs, "trainning loss")
    draw_process("trainning acc", "green", all_train_iters, all_train_accs, "trainning acc")

# Step4、模型评估

# In[14]:


'''
模型评估
'''
# resnet层数定义，要改一下
# train_parameters["layer"] = 50

with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):  # 使用GPU进行训练
    ##with fluid.dygraph.guard():                            #使用CPU进行训练
    model = ResNet("resnet", train_parameters["network_resnet"]["layer"], train_parameters["class_dim"])
    model_dict, _ = fluid.load_dygraph("MyResNet_best")
    # model_dict, _ = fluid.dygraph.load_dygraph("save_dir/ResNet/model_best")
    model.load_dict(model_dict)
    model.eval()

    accs = []
    for batch_id, data in enumerate(eval_reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32').reshape(-1, 3, 224, 224)
        y_data = np.array([x[1] for x in data]).astype('int').reshape(-1, 1)

        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)

        out, acc = model(img, label)
        lab = np.argsort(out.numpy())
        accs.append(acc.numpy()[0])

    avg_acc = np.mean(accs)
    # print(np.mean(accs))
    print("模型校验avg_acc=", avg_acc)

# Step5、模型预测

# In[15]:


import os
import zipfile


def unzip_infer_data(src_path, target_path):
    '''
    解压预测数据集
    '''
    if (not os.path.isdir(target_path)):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))  # HWC to CHW 
    img = img / 255  # 像素值归一化
    return img


infer_src_path = '/home/aistudio/data/data35095/test.zip'
infer_dst_path = '/home/aistudio/data/test'
unzip_infer_data(infer_src_path, infer_dst_path)

# In[16]:


import os

# label_dic = train_parameters['label_dict']

'''
模型预测
'''
with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):  # 使用GPU进行训练
    # with fluid.dygraph.guard():                            #使用CPU进行训练
    model = ResNet("resnet", train_parameters["network_resnet"]["layer"], train_parameters["class_dim"])
    model_dict, _ = fluid.load_dygraph("MyResNet_best")
    # model_dict, _ = fluid.dygraph.load_dygraph("save_dir/ResNet/model_best")
    model.load_dict(model_dict)
    model.eval()

    # 对预测图片进行预处理
    testpath = '/home/aistudio/data/test/testpath.txt'
    with open(testpath, 'r') as f:
        img_list = [line.strip() for line in f]
    infer_path = '/home/aistudio/data/test/test'
    infer_imgs = []
    for imgfn in img_list:
        infer_imag = os.path.join(infer_path, imgfn)
        infer_imgs.append(load_image(infer_imag))
    # print(infer_imgs)
    infer_imgs = np.array(infer_imgs)
    result = []

    for i in range(len(infer_imgs)):
        data = infer_imgs[i]
        dy_x_data = np.array(data).astype('float32')
        dy_x_data = dy_x_data[np.newaxis, :, :, :]
        img = fluid.dygraph.to_variable(dy_x_data)
        out = model(img)
        lab = np.argmax(out.numpy())  # argmax():返回最大数的索引
        result.append(str(lab))


resultpath = '/home/aistudio/data/model_result.txt'
with open(resultpath, 'w') as f:
    f.write("\n".join(result))
print("结束")
