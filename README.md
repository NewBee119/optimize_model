# optimize_model
本项目主要功能是自动完成特征选择与参数调试，用于优化随机森林模型。

## 运行  
Python2.7，需要的依赖库，直接pip安装即可  
运行命令：python ./rf.py   

## 过程  
主要分为两步:一是自动特征选择；二是自动调试参数。  
**自动特征选择**  
1.程序直接从csv文件夹中读取original_data.csv  
2.进入 local_optimal_feature_selection() 函数  
3.采用前向搜索，不断迭代完成特征子集的评估  
4.在代码80行，可自定义设置评价条件  
5.执行完代码，特征重要性可在l_feature_importances.txt中查看  
6.完成特征选择后，新生成的特征文件为feature_selected_data.csv  

**自动调试参数**  
1.程序使用特征选择后的文件feature_selected_data.csv作为输入；  
2.进入automatically_debugging_options()函数  
3.在157行到161行，先计算一次调参前的模型性能  
4.采用贪心法依次对n_estimators，max_features，max_depth，min_samples_split，min_samples_leaf，完成自动调参  
5.在209行到213行，输出调参后的模型性能，以做对比  
  
**手动调参**  
提供手动调参函数：manually_debugging_options()  
在141行手动更改各项参数即可  
  
## 截图  
1.基于不同特征子集的模型迭代过程及局部最优结果输出：     
![image](https://github.com/NewBee119/optimize_model/blob/main/img/fs.jpg)      
  
2.输出降序的特征重要性：   
![image](https://github.com/NewBee119/optimize_model/blob/main/img/fi.jpg)    
  
3.输出自动调参过程：  
![image](https://github.com/NewBee119/optimize_model/blob/main/img/dp.jpg)    

## 说明   
本文的特征文件来自于文献[1]中检测恶意TLS流的部分研究:    
[1] Two-layer detection framework with a high accuracy and efficiency for a malware family over the TLS protocol    
访问链接：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232696  
调参参考这篇技术贴：  
访问链接：https://www.cnblogs.com/jasonfreak/p/5720137.html    
经验总结：  
1. 相比于调参，特征选择的过程更重要；    
2. 采用贪心法自动调参，结果可能还比不上默认参数；  
3. 手动调参，可以辅助确认更优的参数组合；    
下一步：  
1. 可以采用遗传算法，来搜索特征空间，获得全局最优的特征子集；  
2. 也可以进一步探索利用算法来完成调参的过程。  
