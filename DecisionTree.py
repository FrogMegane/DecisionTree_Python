# -*- coding: utf-8 -*-
import csv
import os
import sys
import math
from collections import Counter

'''
    ver1
    class Tree:樹的結構
        left,right:都是None時為leaf
        data:是預測的Label
        condition:是分成兩邊的條件
    
    show:將一顆樹印出
    
    height:計算樹的高度
'''
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.condition=None
    def __repr__(self):
        return repr(self.data)

#印出一顆子樹
def show(node,level=1):
    print(repr(node))
    if type(node) is Tree:
        space='        '*level
        print(space+'R-------',end='')
        show(node.right,level+1)
        print(space+'L-------',end='')
        show(node.left ,level+1)
           
#計算樹的高度
def height(node):
    if node is None:
        return 0
    else:
        lh=height(node.left)
        rh=height(node.right)
        return (lh if lh>=rh else rh)+1
		

#將資料分成兩邊一邊<=val 一邊>val
def splitByVal(D,a,val):
    D1=[]#left
    D2=[]#Right
    for d in D:
        if d[a]<=val:
            D1.append(d)
        else:
            D2.append(d)
    sets=[]
    sets.append(D1)
    sets.append(D2)
    return sets
#找出分割條件
def find_best_split(D,A):
    bsaeEntropy=dataEntropy(D)#用來計算infoGain
    bestInfoGain=0.0
    bestFeature=-100#第幾欄
    bestPartitionVal=0#用多少來分
    for a in A:#所有欄位
        featureList=[d[a] for d in D]#該欄有哪些值
        uniqueVals=set(featureList)#去重複值
        for val in uniqueVals:#用哪個值做分界
            subsets=splitByVal(D,a,val)
            curEntropy=partitionEntropy(subsets)
            infoGain=bsaeEntropy-curEntropy#計算infoGain
            if infoGain>bestInfoGain:#若是infoGain最大
                bestInfoGain=infoGain
                bestFeature=a
                bestPartitionVal=val
    return (bestFeature,bestPartitionVal)

#計算label機率分布
def label_probabilities(labels):
    total_count=len(labels)#分母
    return [count/total_count for count in Counter(labels).values()]#機率s

#計算機率的Entropy
def computeEntropy(probabilities):
    return sum(-p*math.log(p,2)
              for p in probabilities
              if p)#忽略p=0的項

#計算一個Dataset的Entropy
def dataEntropy(datas):
    labels=[d[-1] for d in datas]#將labels取出
    probabilities=label_probabilities(labels)#計算label機率分布
    return computeEntropy(probabilities)#計算出Entropye

#計算一個Subsets的Entropy
def partitionEntropy(subsets):
    total_count=sum([len(subset) for subset in subsets])#分母
    return sum( dataEntropy(subset)*len(subset)/total_count
                for subset in subsets)#每個subset的Entropy乘上權重
    
def LearnTree(D,A):
    labels=[d[-1] for d in D]#將labels取出
    if labels.count(labels[0])==len(labels):          #StopCondition-1 全部都屬於同一種label
        leaf=Tree()
        leaf.data=labels[0]
        return leaf
    elif len(A)==0:                                   #StopCondition-2 沒有剩餘attrib可以供分類
        leaf=Tree()
        leaf.data=Counter(labels).most_common(1)[0][0]#最常出現的1個值，Counter回傳的是一個2維矩陣[(key,count),...]
        return leaf
    else:
        root=Tree()
        field,fieldVal=find_best_split(D,A)#用哪一個field的多少作為分界
        A.remove(field)
        root.condition=lambda d:d[field]<=fieldVal
        root.data='Field%d <= %f ?'%(field,fieldVal)
        D1=[]#left
        D2=[]#Right
        for d in D:
            if root.condition(d):
                D1.append(d)
            else:
                D2.append(d)
        A1=A[:]#因為python call by reference,所以分開
        A2=A[:]
        root.left=LearnTree(D1,A1)
        root.right=LearnTree(D2,A2)
        return root

#使用一訓練好的Decision Tree，分類輸入的資料d
def classify(tree:Tree,d):
    if tree.condition is None:
        return tree.data
    elif tree.condition(d):
        return classify(tree.left,d)
    else:
        return classify(tree.right,d)

#測試正確率
def verify(test):
    correct_count=0
    for t in test:
        predict=classify(tree,t)#分類結果
        ans=t[-1]#答案
        print('predict:%s\tdata:%s'%(predict,t),end='')
        if(predict==ans):
            correct_count=correct_count+1
            print(' OK!')
        else:
            print(' ~~wrong~~')
    return correct_count/len(test)

#刪掉不需要的節點，若是一個分割節點的兩邊都是同一種類型，則將其左右合併到自己上面
def trim(tree):
    if tree.left is None or tree.right is None:
        pass
    else:
        trim(tree.left)
        trim(tree.right)
        if tree.left.data==tree.right.data:
            tree.data=tree.left.data
            tree.condition=None
            tree.left=None
            tree.right=None

			
#data 的前四個欄位是Attributes，第五個欄位是Label
def readCSV(fullpath):
    with open(fullpath,'r') as f:
        reader=csv.reader(f)
        
        #原本都是str，要改成float
        result=[]
        for row in list(reader):
            newRow=[float(row[i]) for i in [0,1,2,3]]
            newRow.append(row[4])
            result.append(newRow)
        return result
    
def checkFileExit(fname):
    if not os.path.isfile(fname):
        print("file does not exist in '%s'"%(fname))
        os.system("pause")
        sys.exit(1)

'''
    training
'''
trainPath=input('請輸入train.csv之路徑(ex: D:/train.csv ) >>>')
checkFileExit(trainPath)

train=readCSV(trainPath)
dataset=train[:]#deep copy
attrib=list(range(len(train[0])-1))#可以用作分類的attributes
tree=LearnTree(dataset,attrib)
show(tree)
print('############################# Original Tree #############################')

trim(tree)
show(tree)
print('############################## After Trim ##############################')

testPath=input('請輸入test.csv之路徑(ex: D:/test.csv ) >>>')
checkFileExit(testPath)

test=readCSV(testPath)
correctness=verify(test)
print('############################### 正確率:%f ###############################'%(correctness))
os.system("pause")
