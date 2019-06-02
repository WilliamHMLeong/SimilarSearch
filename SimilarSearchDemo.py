import numpy as np
import pickle
import pkuseg
import synonyms
from progressbar import *


def load_original_txt(filepath = "./test2.txt"):
    charList = []
    charListAscii = []
    for line in open(filepath,'r', encoding='UTF-8'):           #读入文本
        for i in range(len(line)):
            if ord(line[i]) not in charListAscii:
                charList.append(line[i])
                charListAscii.append(ord(line[i]))

    return charList,charListAscii

def load_txt(filepath = "./test2.txt"):
    f = open(filepath, "r",encoding='UTF-8')

    return f.read()

def load_split_txt(filepath = "./test2.txt"):
    f = open(filepath, "r",encoding='UTF-8')         #读入文本
    text = f.read()
    text = text.split('。')                          #这里进行断句的符号请自己进行选择
    return text


def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def numpy_eye(charList):
    return np.eye(np.array(charList).shape[0])

def string_one_hot(inputs,charDict):
    str_one_hot = []
    for i in range(len(inputs)):
        if inputs[i] not in charDict:
            charDict.update({inputs[i]: len(charDict) + 1})
        str_one_hot.append(charDict[inputs[i]])
    return str_one_hot

def word_dict_one_hot(inputs,wordDict):
    word_one_hot = []
    for i in range(len(inputs)):
        if inputs[i] not in wordDict:
            print(inputs[i])
            print("over")
            wordDict.update({inputs[i]: len(wordDict) + 1})
        word_one_hot.append(wordDict[inputs[i]])
    return word_one_hot

def numpy_one_hot(input1):
    maxnum = max(input1)
    array = np.zeros((maxnum,len(input1)))
    for i in range(len(input1)):
        array[input1[i]-1,i] = 1
    return array


def length_similar(input1,input2):
    if len(input1) > len(input2):
        return float(len(input2)/len(input1))
    else:
        return float(len(input1)) / float(len(input2))

def get_unique(input):                      #取唯一词
    str_unique = []
    widgets = ['Progress get unique: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=10 * len(input)).start()
    for i in range(len(input)):
        pbar.update(10 * i + 1)
        time.sleep(0.0001)
        if input[i] not in str_unique:
            str_unique.append(input[i])
    pbar.finish()
    return str_unique

def char_similar(input1,input2):            #计算字相似度
    base = 0
    input1 = get_unique(input1)
    input2 = get_unique(input2)
    length = max(len(input1),len(input2))
    for i in range(len(input1)):
        if input1[i] in input2:
            base = base + 1
    return(float(base)/float(length))

def char_similar_onehot(input1,input2):
    length = input1.shape[1]+input2.shape[1]
    input1 = np.sum(input1,axis = 1)
    input2 = np.sum(input2,axis = 1)
    print(input1)
    print(input2)
    similar = 1-np.sum(np.abs((input1-input2))/length)
    return similar

def word_similar_onehot(input1,input2):
    length = np.sum(input1)+np.sum(input2)          #计算两个文本的长度
    input1 = np.sum(input1,axis = 1)                #对文本1进行维度压缩，得到该文本的词向量
    input2 = np.sum(input2,axis = 1)                #对文本2进行维度压缩，得到该文本的词向量
    similar = 1-np.sum(np.abs((input1-input2))/length)          #计算词语相似度
    return similar

def order_similar_onehot(input1,input2):
    similar = np.sum(np.abs(input1-input2))
    return similar


def allign_onehot(input1,input2):
    length = max(input1.shape[1], input2.shape[1])
    height = max(input1.shape[0], input2.shape[0])
    if input1.shape[1] < length:
        input1 = np.concatenate((input1,np.zeros((input1.shape[0],length - input1.shape[1]))),axis=1)
    else:
        input2 = np.concatenate((input2,np.zeros((input2.shape[0],length - input2.shape[1]))),axis=1)

    if input1.shape[0] < height:
        input1 = np.concatenate((input1,np.zeros((height - input1.shape[0],input1.shape[1]))),axis=0)
    else:
        input2 = np.concatenate((input2,np.zeros((height - input2.shape[0],input2.shape[1]))),axis=0)

    return input1,input2


def build_dict(charList):
    charDict = {}
    charList_onehot = numpy_eye(charList)

    widgets = ['Progress build dict: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=10 * len(charList)).start()

    for i in range(len(charList)):
        pbar.update(10 * i + 1)
        time.sleep(0.0001)
        charDict.update({charList[i]: np.argmax(charList_onehot,axis = 0)[i]})

    pbar.finish()
    return  charDict

def build_word_dict(strList):
    wordDict = {}
    widgets = ['Progress bulid word dict: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=10 * len(strList)).start()
    for i in range(len(strList)):
        pbar.update(10*i + 1)
        time.sleep(0.0001)
        for j in range(len(seg.cut(strList[i]))):
            if seg.cut(strList[i])[j] not in wordDict:
                wordDict.update({seg.cut(strList[i])[j]: len(wordDict)+1})
    pbar.finish()
    return wordDict


def build_word_index(wordList,wordDict):
    widgets = ['Progress build str index: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    max1 = len(wordDict)
    array = np.zeros((max1, len(wordList)))
    pbar = ProgressBar(widgets=widgets, maxval=10 * len(wordList)).start()
    for i in range(len(wordList)):
        pbar.update(10 * i + 1)
        time.sleep(0.0001)
        tempseg = seg.cut(wordList[i])
        str_one_hot = word_dict_one_hot(tempseg,wordDict)
        for j in range(len(str_one_hot)):
            array[str_one_hot[j]-1,i] = 1
    pbar.finish()
    return array

def get_syn_word(input1):
    str_onw_syn_list = []

    for i in range(len(input1)):
        str_onw_syn_list.append(synonyms.nearby(input1[i]))

    return str_onw_syn_list

def find_syn_word(input,dict,word_index):
    index = []
    syn = []
    words = []
    for i in range(len(input)):
        for j in range(len(input[i][0])):
            if str(input[i][0][j]) in dict:
                for k in range(int(np.sum(word_index,axis=1)[dict[str(input[i][0][j])]-1])):
                    index.append(np.argmax(word_index[dict[str(input[i][0][j])]-1],axis=0))
                    word_index[dict[str(input[i][0][j])]-1,np.argmax(word_index[dict[str(input[i][0][j])]-1])] = 0
                    syn.append(input[i][1][j]/len(input))
                    words.append(str(input[i][0][j]))

    return index,syn,words

seg = pkuseg.pkuseg()


wordList = load_split_txt()
# wordDict = build_word_dict(wordList)
# save_obj(wordDict,"chiWordDict")

wordDict = load_obj("chiWordDict")
print(wordDict)

# word_index = build_word_index(wordList,wordDict)
# save_obj(word_index,"word_index")
word_index = load_obj("word_index")
str_one = '灵魂'

str_one = seg.cut(str_one)

print(wordDict['灵魂'])

str_one_syn_list = get_syn_word(str_one)

print(str_one)

index,syn,words = find_syn_word(str_one_syn_list,wordDict,word_index)

print(index)

str_one_hot1 = word_dict_one_hot(get_unique(seg.cut("一人是撒一")),wordDict)
str_one_hot2 = word_dict_one_hot(get_unique(seg.cut("灵魂")),wordDict)


str_one_hot1 = numpy_one_hot(str_one_hot1)
str_one_hot2 = numpy_one_hot(str_one_hot2)

str_one_hot1,str_one_hot2 = allign_onehot(str_one_hot1,str_one_hot2)

cs = word_similar_onehot(str_one_hot1,str_one_hot2)


ors = order_similar_onehot(str_one_hot1,str_one_hot2)

print(cs)
print(ors)

for i in range(len(index)):
    print(wordList[index[i]])