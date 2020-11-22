#!/usr/bin/env python
# coding: utf-8

# ## 作业描述：
# 
# ### **✓代码跑通**
# ### 请大家根据课上所学内容，构建图像分类Fine-tune模型并跑通，在此基础上可尝试调参优化。
# ### ✓**调优**
# ### 思考并动手进行调优，以在测试集上的acc为评价指标，测试集上acc越高，得分越高！
# ### **✓自定义数据集**
# ### 鼓励老师自己找数据集，使用自定义数据集完成finetune
# 
# 

# # PaddleHub之图像分类

# ## 一、任务简介
# 
# 
# ### 图像分类，顾名思义，是一个输入图像，输出对该图像内容分类的描述的问题。它是计算机视觉的核心，实际应用广泛。
# 
# ### 本次实践将会使用PaddleHub  Finetune API来完成图像分类的深度迁移学习。
# 
# 
# <div  align="center">   
# <img src="https://ai-studio-static-online.cdn.bcebos.com/7578df1c689640d380170a112da1193f30441b44f02e47bca553e3b52c86896c" width = "500" height = "400" align=center />
# </div>
# 
# 
# <br/>
# 

# In[ ]:


#CPU环境启动请务必执行该指令
#%set_env CPU_NUM=1 


# In[1]:


#安装paddlehub
get_ipython().system('pip install paddlehub==1.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple')


# ## 二、任务实践
# ### Step1、基础工作
# 
# 加载数据文件
# 
# 导入python包

# In[2]:


import os
import zipfile

'''
参数配置
'''
train_parameters = {
    "src_path":"/home/aistudio/data/data43082/sisters.zip",      #原始数据集路径
    "target_path":"/home/aistudio/data/",                         #要解压的路径
    "train_list_path": "/home/aistudio/data/sisters/train.txt",   #train.txt路径
    "eval_list_path": "/home/aistudio/data/sisters/eval.txt",     #eval.txt路径
    "num_epochs": 10,                                              #训练轮数
    "train_batch_size": 8,                                        #训练时每个批次的大小
    "checkpoint_path": "cv_finetune_turtorial_demo",
    "eval_interval": 10
   
}

def unzip_data(src_path,target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至data目录下
    '''
    if(not os.path.isdir(target_path + "sisters")):     
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()

src_path = train_parameters["src_path"]
target_path = train_parameters["target_path"]
unzip_data(src_path,target_path)



# ### Step2、加载预训练模型
# 
# 接下来我们要在PaddleHub中选择合适的预训练模型来Finetune，由于是图像分类任务，因此我们使用经典的ResNet-50作为预训练模型。PaddleHub提供了丰富的图像分类预训练模型，包括了最新的神经网络架构搜索类的PNASNet，我们推荐您尝试不同的预训练模型来获得更好的性能。

# In[3]:


import paddlehub as hub
module = hub.Module(name="resnet_v2_101_imagenet")


# ### Step3、数据准备
# 
# 接着需要加载图片数据集。我们使用自定义的数据进行体验，请查看[适配自定义数据](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub适配自定义数据完成FineTune)

# In[4]:


from paddlehub.dataset.base_cv_dataset import BaseCVDataset
   
class DemoDataset(BaseCVDataset):	
   def __init__(self):	
       # 数据集存放位置
       
       self.dataset_dir = "data/sisters"
       super(DemoDataset, self).__init__(
           base_path=self.dataset_dir,
           train_list_file="train.txt",
           validate_list_file="eval.txt",
           test_list_file="eval.txt",
           label_list_file="label.txt",
           )
dataset = DemoDataset()


# ### Step4、生成数据读取器
# 
# 接着生成一个图像分类的reader，reader负责将dataset的数据进行预处理，接着以特定格式组织并输入给模型进行训练。
# 
# 当我们生成一个图像分类的reader时，需要指定输入图片的大小

# ## ####请在以下cell中补充代码，构造数据提供器#####

# In[7]:


data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)


# ### Step5、配置策略
# 在进行Finetune前，我们可以设置一些运行时的配置，例如如下代码中的配置，表示：
# 
# * `use_cuda`：设置为False表示使用CPU进行训练。如果您本机支持GPU，且安装的是GPU版本的PaddlePaddle，我们建议您将这个选项设置为True；
# 
# * `epoch`：迭代轮数；
# 
# * `batch_size`：每次训练的时候，给模型输入的每批数据大小为32，模型训练时能够并行处理批数据，因此batch_size越大，训练的效率越高，但是同时带来了内存的负荷，过大的batch_size可能导致内存不足而无法训练，因此选择一个合适的batch_size是很重要的一步；
# 
# * `log_interval`：每隔10 step打印一次训练日志；
# 
# * `eval_interval`：每隔50 step在验证集上进行一次性能评估；
# 
# * `checkpoint_dir`：将训练的参数和数据保存到cv_finetune_turtorial_demo目录中；
# 
# * `strategy`：使用DefaultFinetuneStrategy策略进行finetune；
# 
# 更多运行配置，请查看[RunConfig](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-RunConfig)
# 
# 同时PaddleHub提供了许多优化策略，如`AdamWeightDecayStrategy`、`ULMFiTStrategy`、`DefaultFinetuneStrategy`等，详细信息参见[策略](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Strategy)

# ## ####请在以下cell中补充代码，配置finetune策略#####

# In[10]:


config = hub.RunConfig(
    use_cuda=True,
    num_epoch=1,
    checkpoint_dir="cv_finetune_turtorial_demo",
    batch_size=3,
    eval_interval=10,
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy()) 


# ### Step6、组建Finetune Task
# 有了合适的预训练模型和准备要迁移的数据集后，我们开始组建一个Task。
# 
# 由于该数据设置是一个二分类的任务，而我们下载的分类module是在ImageNet数据集上训练的千分类模型，所以我们需要对模型进行简单的微调，把模型改造为一个二分类模型：
# 
# 1. 获取module的上下文环境，包括输入和输出的变量，以及Paddle Program；
# 2. 从输出变量中找到特征图提取层feature_map；
# 3. 在feature_map后面接入一个全连接层，生成Task；

# In[11]:


input_dict, output_dict, program = module.context(trainable=True)
img = input_dict["image"]
feature_map = output_dict["feature_map"]
feed_list = [img.name]

task = hub.ImageClassifierTask(
    data_reader=data_reader,
    feed_list=feed_list,
    feature=feature_map,
    num_classes=dataset.num_labels,
    config=config)


# ### Step7、开始Finetune
# 
# 我们选择`finetune_and_eval`接口来进行模型训练，这个接口在finetune的过程中，会周期性的进行模型效果的评估，以便我们了解整个训练过程的性能变化。

# In[12]:


run_states = task.finetune_and_eval()


# ### Step8、预测
# 
# 当Finetune完成后，我们使用模型来进行预测，先通过以下命令来获取测试的图片

# In[13]:


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as image
#显示matplotlib生成的图形
get_ipython().run_line_magic('matplotlib', 'inline')
def show_image(image_path):
    '''
    展示图片
    '''
    img = image.imread(image_path)
    plt.figure(figsize=(10,10))
    plt.imshow(img) 
    plt.axis('off') 
    plt.show()

index = 0
data = ["data/sisters/infer/infer_ningjing.jpg"]

label_map = dataset.label_dict()
run_states = task.predict(data=data)
results = [run_state.run_results for run_state in run_states]

for batch_result in results:
    batch_result = np.argmax(batch_result, axis=2)[0]
    for result in batch_result:
        index += 1
        result = label_map[result]
        print("input %i is %s, and the predict result is %s" %
              (index, data[index - 1], result))

show_image(data[0])

