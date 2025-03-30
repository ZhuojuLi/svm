# svm
SVM 是一种 最大间隔分类器（Maximum Margin Classifier），它的目标不是只要分对，而是要：

找到一个分隔超平面，使得离它最近的样本（支持向量）尽可能远离它。
![image](https://github.com/user-attachments/assets/08220c74-a5fe-4f3f-88d1-c7d3bbcee8ea)
![image](https://github.com/user-attachments/assets/ce547331-d15b-41a2-b8e0-e3844f72fb7a)
## 目标函数（最大化间隔）
![image](https://github.com/user-attachments/assets/2bb42337-7f1d-4649-93b5-181a6a52be00)
![image](https://github.com/user-attachments/assets/4f6f7531-39a8-41f9-87d8-1904fc49fb3b)
![image](https://github.com/user-attachments/assets/3b5b8cba-a572-41ce-b05d-76c00ef2bee4)
## 满足$α_i>0$的点就是支持向量，只有它们影响决策边界。
