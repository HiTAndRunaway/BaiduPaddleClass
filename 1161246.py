#!/usr/bin/env python
# coding: utf-8

# ## 作业描述：
# ## 请在最下方提示位置补充代码，东方卫视和浙江卫视的《平凡的荣耀》收视率变化趋势进行绘制，如下图所示：
# 
# <img  style="height:400px;weight:600px"  src="https://ai-studio-static-online.cdn.bcebos.com/38d052ffb305467d891609b5e12eec0594685a89a7064767bcf9f4263b15b589">
# 

# ## 任务描述
# 
# 
# 
# ### 本次实践使用Python来爬取百度百科中《平凡的荣耀》所有演员的信息，以及收视率，并进行可视化分析。
# 
# ### 数据获取：https://baike.baidu.com/item/平凡的荣耀
# 
# <br/>
# <br/>
# 
# 
# <img src ="https://ai-studio-static-online.cdn.bcebos.com/04d2190a3751433fb28319921ce6d6e85cc3684e8e004ab58fb967614e0b6e91" height='500' width='500'/>
# <img src="https://ai-studio-static-online.cdn.bcebos.com/99e285cacd104872963577cf17a0dc3266e731fff25f46abac5734889b311561" height='500' width='500' />
# 
# 
# <br/>
# <br/>
# 

# <br/>
# 
# **上网的全过程:**
# 
#     普通用户:
# 
#     打开浏览器 --> 往目标站点发送请求 --> 接收响应数据 --> 渲染到页面上。
# 
#     爬虫程序:
# 
#     模拟浏览器 --> 往目标站点发送请求 --> 接收响应数据 --> 提取有用的数据 --> 保存到本地/数据库。
# 
# 
# **爬虫的过程**：
# 
#     1.发送请求（requests模块）
# 
#     2.获取响应数据（服务器返回）
# 
#     3.解析并提取数据（BeautifulSoup查找或者re正则）
# 
#     4.保存数据
# 
# 

# 
# <br/>
# 
# **本实践中将会使用以下两个模块，首先对这两个模块简单了解以下：**

# <br/>
# 
# **request模块：**
# 
#     requests是python实现的简单易用的HTTP库，官网地址：http://cn.python-requests.org/zh_CN/latest/
#     
#     requests.get(url)可以发送一个http get请求，返回服务器响应内容。
#     
#     
# 
# 
# 
# 

# <br/>
# 
# **BeautifulSoup库：**
# 
#     BeautifulSoup 是一个可以从HTML或XML文件中提取数据的Python库。网址：https://beautifulsoup.readthedocs.io/zh_CN/v4.4.0/
#     
#     BeautifulSoup支持Python标准库中的HTML解析器,还支持一些第三方的解析器,其中一个是 lxml。
#     
#     BeautifulSoup(markup, "html.parser")或者BeautifulSoup(markup, "lxml")，推荐使用lxml作为解析器,因为效率更高。

# In[1]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')
get_ipython().system('pip install lxml -t /home/aistudio/external-libraries')


# In[2]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
import sys
sys.path.append('/home/aistudio/external-libraries')


# ## 数据爬取
# ## 一、爬取百度百科中《平凡的荣耀》中所有演员信息，返回页面数据

# In[3]:


import json
import re
import requests
import datetime
from bs4 import BeautifulSoup
import os


def crawl_wiki_data():
    """
    爬取百度百科中《平凡的荣耀》中演员信息，返回html
    """
    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    url='https://baike.baidu.com/item/平凡的荣耀'                         

    try:
        response = requests.get(url,headers=headers)
        #将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象, 可以传入一段字符串
        soup = BeautifulSoup(response.text,'lxml')           
        #返回class="lemmaWgt-roleIntroduction"的div,即“角色介绍”下方的div
        roleIntroductions = soup.find('div',{'class':'lemmaWgt-roleIntroduction'})
        all_roleIntroductions = roleIntroductions.find_all('li')
        actors = []
        for every_roleIntroduction in all_roleIntroductions:
             actor = {}    
             if every_roleIntroduction.find('div',{'class':'role-actor'}):
                 #找演员名称和演员百度百科连接               
                actor["name"] = every_roleIntroduction.find('div',{'class':'role-actor'}).find('a').text
                actor['link'] =  'https://baike.baidu.com' + every_roleIntroduction.find('div',{'class':'role-actor'}).find('a').get('href')
             actors.append(actor)
    except Exception as e:
        print(e)

    json_data = json.loads(str(actors).replace("\'","\""))   
    with open('work/' + 'actors.json', 'w', encoding='UTF-8') as f:
        json.dump(json_data, f, ensure_ascii=False)



# ## 二、爬取每个演员的百度百科页面的信息，并进行保存

# In[4]:


def crawl_everyone_wiki_urls():
    '''
    爬取每个演员的百度百科图片，并保存
    ''' 
    with open('work/' + 'actors.json', 'r', encoding='UTF-8') as file:
         json_array = json.loads(file.read())
    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36' 
     }  
    actor_infos = []
    for star in json_array:
        actor_info = {}       
        name = star['name']
        link = star['link']
        actor_info['name'] = name
        #向演员个人百度百科发送一个http get请求
        response = requests.get(link,headers=headers)        
        #将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象
        bs = BeautifulSoup(response.text,'lxml')       
        #获取演员的民族、星座、血型、体重等信息
        base_info_div = bs.find('div',{'class':'basic-info cmn-clearfix'})
        dls = base_info_div.find_all('dl')
        for dl in dls:
            dts = dl.find_all('dt')
            for dt in dts:
                if "".join(str(dt.text).split()) == '民族':
                     actor_info['nation'] = dt.find_next('dd').text
                if "".join(str(dt.text).split()) == '星座':
                     actor_info['constellation'] = dt.find_next('dd').text
                if "".join(str(dt.text).split()) == '血型':  
                     actor_info['blood_type'] = dt.find_next('dd').text
                if "".join(str(dt.text).split()) == '身高':  
                     height_str = str(dt.find_next('dd').text)
                     actor_info['height'] = str(height_str[0:height_str.rfind('cm')]).replace("\n","")
                if "".join(str(dt.text).split()) == '体重':  
                     actor_info['weight'] = str(dt.find_next('dd').text).replace("\n","")
                if "".join(str(dt.text).split()) == '出生日期':  
                     birth_day_str = str(dt.find_next('dd').text).replace("\n","")
                     if '年' in  birth_day_str:
                         actor_info['birth_day'] = birth_day_str[0:birth_day_str.rfind('年')]
        actor_infos.append(actor_info) 
        #将演员个人信息存储到json文件中
        json_data = json.loads(str(actor_infos).replace("\'","\""))   
        with open('work/' + 'actor_infos.json', 'w', encoding='UTF-8') as f:
            json.dump(json_data, f, ensure_ascii=False)

     


# ## 三、爬取《平凡的荣耀》的收视情况，并保存

# In[5]:


def crawl_viewing_data():
    """
    爬取百度百科中《平凡的荣耀》收视情况，返回html
    """
    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    url='https://baike.baidu.com/item/平凡的荣耀'                         

    try:
        response = requests.get(url,headers=headers)
        #将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象, 可以传入一段字符串
        soup = BeautifulSoup(response.text,'lxml')     
          
        #返回所有的<table>所有标签
        tables = soup.find_all('table')
        crawl_table_title = "收视情况"
        for table in  tables:           
            #对当前节点前面的标签和字符串进行查找
            table_titles = table.find_previous('div')
            for title in table_titles:
                if(crawl_table_title in title):
                    return table       
    except Exception as e:
        print(e)



# In[6]:




def parse_viewing_data(viewing_table):
    """
    对《平凡的荣耀》的收视情况table进行解析，并保存
    """
    viewing_datas = []

    trs = viewing_table.find_all('tr')

    for i in range(len(trs)):
        if i < 2:
            continue
        viewing_data = {}
        tds = trs[i].find_all('td')
        viewing_data["broadcastDate"]= tds[0].text
        viewing_data["dongfang_rating"]= tds[1].text
        viewing_data["dongfang_rating_share"]= tds[2].text
        viewing_data["dongfang_ranking"]= tds[3].text
        viewing_data["zhejiang_rating"]= tds[4].text
        viewing_data["zhejiang_rating_share"]= tds[5].text
        viewing_data["zhejiang_ranking"]= tds[6].text
        viewing_datas.append(viewing_data)
    #将个人信息存储到json文件中
    json_data = json.loads(str(viewing_datas).replace("\'","\""))   
    with open('work/' + 'viewing_infos.json', 'w', encoding='UTF-8') as f:
        json.dump(json_data, f, ensure_ascii=False)



# ## 四、数据爬取主程序

# In[7]:


if __name__ == '__main__':

     #爬取百度百科中《平凡的荣耀》中演员信息，返回html
     html = crawl_wiki_data()

     #爬取每个演员的信息,并保存
     crawl_everyone_wiki_urls()
     
     #1、爬取百度百科中《平凡的荣耀》收视情况，返回html   2、对《平凡的荣耀》的收视情况table进行解析，并保存
     viewing_table = crawl_viewing_data()
     parse_viewing_data(viewing_table)
     
     print("所有信息爬取完成！")

     


# # 数据分析

# In[8]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install matplotlib -t /home/aistudio/external-libraries')


# In[9]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
import sys
sys.path.append('/home/aistudio/external-libraries')


# In[10]:


# 下载中文字体
#!wget https://mydueros.cdn.bcebos.com/font/simhei.ttf
# 将字体文件复制到matplotlib字体路径
get_ipython().system('cp /home/aistudio/work/simhei.ttf /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/')
# 创建系统字体文件路径
get_ipython().system('mkdir .fonts')
# 复制文件到该路径
get_ipython().system('cp  /home/aistudio/work/simhei.ttf  .fonts/')
get_ipython().system('rm -rf .cache/matplotlib')


# ## 一、 绘制东方卫视收视率变化趋势

# In[11]:



import matplotlib.pyplot as plt
import numpy as np 
import json
import matplotlib.font_manager as font_manager
import pandas as pd
#显示matplotlib生成的图形
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_json('work/viewing_infos.json',dtype = {'broadcastDate' : str})
#print(df)

broadcastDate_list = df['broadcastDate']
dongfang_rating_list = df['dongfang_rating']

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.figure(figsize=(15,8))
plt.title("《平凡的荣耀》东方卫视收视率变化趋势",fontsize=20) 
plt.xlabel("播出日期",fontsize=20) 
plt.ylabel("收视率%",fontsize=20) 
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)
plt.plot(broadcastDate_list,dongfang_rating_list) 
plt.grid() 
plt.savefig('/home/aistudio/work/reuslt01.jpg')
plt.show()


# ##  #####请在下面cell中对东方卫视和浙江卫视的《平凡的荣耀》收视率变化趋势进行绘制，并进行分析#####

# 趋势：周末收视情况不错，收视起伏比较大。
