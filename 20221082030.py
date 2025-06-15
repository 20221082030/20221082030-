import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

def parse_output(output_str):
    """解析输出字符串为四元组列表"""
    quads = []
    parts = output_str.split('[SEP]')  # 用分隔符分割不同四元组
    for part in parts:
        part = part.strip()
        if not part or '[END]' not in part:  # 跳过空部分或没有结束标记的部分
            continue
        quad_str = part.split('[END]')[0].strip()  # 提取四元组字符串
        components = [c.strip() for c in quad_str.split('|')]  # 分割四个组件
        if len(components) == 4:  
            quads.append(tuple(components))
    return quads

def prepare_data(train_data):
    """准备训练数据，包括序列标注和分类任务"""
    X_seq, y_seq = [], []  # 序列标注的输入和标签
    X_class, y_group, y_hate = [], [], []  # 分类任务的输入和标签
    
    for item in train_data:
        content = item['content']  # 文本内容
        output = item['output']  # 标注输出
        quads = parse_output(output)  # 解析为四元组
        
        if not quads:  
            continue
            
        # 准备序列标注数据
        chars = list(content)  # 将文本转为字符列表
        labels = ['O'] * len(chars)  # 初始化标签为'O'(其他)
        
        for quad in quads:
            target, argument, _, _ = quad  # 解构四元组
            
            # 标注目标(target)
            if target != "NULL":
                start = content.find(target)  # 查找目标在文本中的位置
                if start != -1:
                    end = start + len(target)
                    for i in range(start, end):
                        prefix = 'B-T' if i == start else 'I-T'  # 开始用B-T，其余用I-T
                        if labels[i] == 'O':  # 只覆盖'O'标签
                            labels[i] = prefix
            
            # 标注论点(argument)
            start = content.find(argument)
            if start != -1:
                end = start + len(argument)
                for i in range(start, end):
                    prefix = 'B-A' if i == start else 'I-A'  # 开始用B-A，其余用I-A
                    if labels[i] == 'O':
                        labels[i] = prefix
        
        X_seq.append(chars)  # 添加字符序列
        y_seq.append(labels)  # 添加对应标签
        
        # 准备分类数据
        for quad in quads:
            target, argument, group, hate = quad  # 解构四元组
            text = f"{target} {argument}" if target != "NULL" else argument  # 组合文本
            X_class.append(text)  # 添加分类文本
            y_group.append(group)  # 添加群体标签
            y_hate.append(hate)  # 添加仇恨言论标签
    
    return X_seq, y_seq, X_class, y_group, y_hate

def word2features(sent, i):
    """为每个字符生成特征"""
    word = sent[i]  # 当前字符
    
    features = {
        'bias': 1.0,  # 偏置项
        'word': word,  # 字符本身
        'word.lower()': word.lower(),  # 小写形式
        'word.isdigit()': word.isdigit(),  # 是否数字
        'word.ispunct()': word in ",.?!;:'\"",  # 是否标点
    }
    
    # 上下文特征
    if i > 0:  # 前一个字符特征
        prev_word = sent[i-1]
        features.update({
            '-1:word': prev_word,
            '-1:isdigit': prev_word.isdigit(),
        })
    else:
        features['BOS'] = True  # 文本开始标记
        
    if i < len(sent)-1:  # 后一个字符特征
        next_word = sent[i+1]
        features.update({
            '+1:word': next_word,
            '+1:isdigit': next_word.isdigit(),
        })
    else:
        features['EOS'] = True  # 文本结束标记
        
    return features

def sent2features(sent):
    """将整个句子转为特征列表"""
    return [word2features(sent, i) for i in range(len(sent))]

def extract_components(chars, labels):
    """从标注序列中提取目标和论点"""
    targets, arguments = [], []  # 存储提取的结果
    current_target, current_arg = [], []  # 当前正在处理的目标/论点
    in_target, in_arg = False, False  
    
    for char, label in zip(chars, labels):
        if label == 'B-T':  # 目标开始
            if current_target:
                targets.append(''.join(current_target))
            current_target = [char]
            in_target = True
        elif label == 'I-T' and in_target:  # 目标中间部分
            current_target.append(char)
        elif label == 'B-A':  # 论点开始
            if current_arg:
                arguments.append(''.join(current_arg))
            current_arg = [char]
            in_arg = True
        elif label == 'I-A' and in_arg:  # 论点中间部分
            current_arg.append(char)
        else:  # 其他情况
            if current_target:  # 完成当前目标
                targets.append(''.join(current_target))
                current_target = []
                in_target = False
            if current_arg:  # 完成当前论点
                arguments.append(''.join(current_arg))
                current_arg = []
                in_arg = False
    
    # 添加剩余部分
    if current_target:
        targets.append(''.join(current_target))
    if current_arg:
        arguments.append(''.join(current_arg))
    
    # 如果没有找到论点，使用整个文本作为默认论点
    if not arguments:
        arguments.append(''.join(chars))
    
    return targets, arguments

# 加载训练数据
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 准备数据
X_seq, y_seq, X_class, y_group, y_hate = prepare_data(train_data)

# 训练CRF模型进行序列标注
crf = CRF(
    algorithm='lbfgs',  # 使用L-BFGS优化算法
    c1=0.1,  # L1正则化系数
    c2=0.1,  # L2正则化系数
    max_iterations=100,  # 最大迭代次数
    all_possible_transitions=True  # 允许所有可能的转移
)
X_seq_features = [sent2features(sent) for sent in X_seq]  # 转换特征
crf.fit(X_seq_features, y_seq)  # 训练模型

# 训练分类器
group_encoder = LabelEncoder()  # 群体标签编码器
y_group_encoded = group_encoder.fit_transform(y_group)  # 编码群体标签

hate_encoder = LabelEncoder()  # 仇恨言论标签编码器
y_hate_encoded = hate_encoder.fit_transform(y_hate)  # 编码仇恨言论标签

# 使用字符n-gram更好地表示中文文本
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))

group_clf = Pipeline([
    ('tfidf', vectorizer),  # 文本向量化
    ('clf', LinearSVC())  # 线性支持向量机分类器
])
group_clf.fit(X_class, y_group_encoded)  # 训练群体分类器

hate_clf = Pipeline([
    ('tfidf', vectorizer),
    ('clf', LinearSVC())
])
hate_clf.fit(X_class, y_hate_encoded)  # 训练仇恨言论分类器

# 加载测试数据
with open('test1.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 处理测试数据
results = []
for item in test_data:
    content = item['content']  # 测试文本内容
    chars = list(content)  # 转为字符列表
    
    # 预测标签序列
    features = sent2features(chars)
    labels = crf.predict_single(features)
    
    # 提取目标和论点
    targets, arguments = extract_components(chars, labels)
    
    # 处理没有找到目标的情况
    if not targets:
        targets = ["NULL"]  # 使用NULL作为默认目标
    
    # 生成四元组
    quads = []
    for target in targets[:1]:  
        for argument in arguments[:1]:  
            text = f"{target} {argument}" if target != "NULL" else argument
            
            # 预测群体和仇恨言论类型
            group_encoded = group_clf.predict([text])[0]
            group = group_encoder.inverse_transform([group_encoded])[0]
            
            hate_encoded = hate_clf.predict([text])[0]
            hate = hate_encoder.inverse_transform([hate_encoded])[0]
            
            quads.append(f"{target} | {argument} | {group} | {hate}")
    
    # 格式化输出
    if not quads:  # 如果没有生成任何四元组
        quads.append("NULL | NULL | non-hate | non-hate")  # 使用默认四元组
    
    output = " [SEP] ".join(quads) + " [END]"  # 用分隔符连接多个四元组
    results.append(output)

# 保存结果
with open('demo.txt', 'w', encoding='utf-8') as f:
    for res in results:
        f.write(res + '\n')

print("结果已保存到demo.txt")
