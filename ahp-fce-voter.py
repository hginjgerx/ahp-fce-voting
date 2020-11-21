from scipy.special import comb
import numpy
import random
import time

exe_all = 10
exe_num = 7

# 记录各执行体被采纳为表决结果的次数
adopted_time = [0] * exe_all

# 记录各执行体的输出结果
out = [None] * exe_all

# 记录各执行体被选择为在线的次数
on_time = [0] * exe_all

exe = [None] * exe_all

# 各执行体的输出结果集合，期望的正确输出结果为1
exe[0] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4,
          5, 5, 5, 5, 5, 5)

exe[1] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
          4, 5, 5, 5, 5, 5)

exe[2] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5,
          5, 5, 5, 5, 5, 5)

exe[3] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
          4, 4, 4, 4, 5, 5, 5, 5, 5)

exe[4] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
          4, 4, 4, 4, 4, 4, 5, 5, 5)

exe[5] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
          4, 4, 4, 4, 4, 5, 5, 5, 5)

exe[6] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4,
          4, 5, 5, 5, 5, 5, 5, 5, 5)

exe[7] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4,
          5, 5, 5, 5, 5)

exe[8] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4,
          5, 5, 5, 5, 5)

exe[9] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,
          4, 4, 5, 5, 5)
# 各执行体之间的异构度矩阵；异构度范围为0-1分，0分为完全相同，1分为完全异构
# 矩阵应为对称矩阵，主对角线为全0
he_all = [None] * 10
for i in range(10):
    he_all[i] = [0] * 10
# he_all = ((0,    0.33, 0.78, 0.6,  0.17, 0.71, 0.5,  0.64, 1,    0.67),
#          (0.33, 0,    0.56, 0.47, 0.33, 0.57, 0.17, 0.45, 0.56, 0.33),
#          (0.78, 0.56, 0,    0.5,  0.78, 0.64, 0.33, 0.25, 0.33, 0.33),
#          (0.6,  0.47, 0.5,  0,    0.47, 0.29, 0.33, 0.43, 0.5,  0.2),
#          (0.17, 0.33, 0.78, 0.47, 0,    0.71, 0.5,  0.64, 1,    0.67),
#          (0.71, 0.57, 0.64, 0.29, 0.71, 0,    0.43, 0.54, 0.45, 0.29),
#          (0.5,  0.17, 0.33, 0.33, 0.5,  0.43, 0,    0.27, 0.33, 0.17),
#          (0.64, 0.45, 0.25, 0.43, 0.64, 0.54, 0.27, 0,    0.5,  0.27),
#          (1,    0.56, 0.33, 0.5,  1,    0.45, 0.33, 0.5,  0,    0.33),
#          (0.67, 0.33, 0.33, 0.2,  0.67, 0.29, 0.17, 0.27, 0.33, 0))


# 执行体输出结果的类
class OutputSet:
    # 初始化函数，参数为创建本对象的执行体编号
    def __init__(self, num):
        # 定义输出了该结果的执行体编号列表
        self.__assentor = []
        self.__assentor.append(num)
        # 定义该输出结果的归一化一致度、归一化平均历史置信度、平均异构度
        self.__cs = self.__cf = self.__he = 0

    # 计算归一化一致度，参数为参与表决的执行体数量
    def __consistency(self):
        # 归一化一致度 = 输出该结果的的执行体数量 / 参与表决的执行体数量
        self.__cs = len(self.__assentor) / exe_num

    # 计算平均历史置信度
    def __confidence(self):
        L = len(self.__assentor)
        for i in range(L):
            # 每个执行体的置信度 = 采纳次数 / 在线次数
            self.__cf += adopted_time[self.__assentor[i]] / on_time[self.__assentor[i]]
        # 平均历史置信度 = 输出该结果的执行体置信度之和 / 输出该结果的执行体数量
        self.__cf = self.__cf / L

    # 计算平均异构度
    def __heterogeneity(self):
        he_sum = 0
        L = len(self.__assentor)
        for i in range(L):
            for j in range(i + 1, L, 1):
                he_sum += he_all[self.__assentor[i]][self.__assentor[j]]
        if L >= 2:
            # 平均异构度 = 输出该结果的执行体所有两两组合的异构度之和 / 输出该结果的执行体所有两两组合的数量
            self.__he = he_sum / comb(L, 2)
        else:
            self.__he = 0

    # 记录输出了该结果的执行体编号。参数为需要添加到assentor列表的执行体编号
    def assentor_append(self, num):
        # 检测添加的编号是否合法
        if num < exe_all:
            self.__assentor.append(num)

    # 查询读取assentor列表。参数为列表索引
    # 若查询的索引合法，则返回相应的执行体编号，否则返回None
    def assentor_read(self, num):
        # 检测查询的索引是否合法
        if num < len(self.__assentor):
            return self.__assentor[num]
        else:
            return None

    # 查询读取一致度
    def cs_read(self):
        return self.__cs

    # 查询读取置信度
    def cf_read(self):
        return self.__cf

    # 查询读取异构度
    def he_read(self):
        return self.__he

    # 构造评价矩阵并进行评分。
    # 参数weight：3个准则的权重向量
    # 返回隶属于等级高的评分
    def evaluation(self, weight):
        # 定义3行2列评价矩阵，其中3是评价准则数量（一致度、历史置信度、异构度），2是评价等级数量（高、低）
        eval_matrix = numpy.array([[0, 0],
                                   [0, 0],
                                   [0, 0]], dtype=numpy.float32)
        # 计算归一化一致度、归一化平均历史置信度、平均异构度
        self.__consistency()
        self.__confidence()
        self.__heterogeneity()
        # 构造评价矩阵
        eval_matrix[0][0] = self.__cs
        eval_matrix[0][1] = 1 - self.__cs
        eval_matrix[1][0] = self.__cf
        eval_matrix[1][1] = 1 - self.__cf
        eval_matrix[2][0] = self.__he
        eval_matrix[2][1] = 1 - self.__he
        # 计算评价结果向量
        eval_vector = numpy.dot(weight, eval_matrix)
        # 根据最大隶属度原则，返回隶属于等级高的评分
        return eval_vector[0]


# 层次分析法AHP。返回权重向量
def ahp():
    # 判断矩阵，用于量化一致度、历史置信度、异构度三个准则的相对重要性
    # 若准则i相比准则j同等重要，judge[i][j]=1；若准则i相比准则j稍微重要，judge[i][j]=3；
    # 若准则i相比准则j明显重要，judge[i][j]=5；若准则i相比准则j强烈重要，judge[i][j]=7；
    # 若准则i相比准则j极端重要，judge[i][j]=9；2,4,6,8为相邻判断的中间值；令judge[j][i]=1/judge[i][j]
    judge = ((1,   2,   3),
             (1/2, 1,   1.5),
             (1/3, 1/1.5, 1))
    # 计算判断矩阵的特征值、特征向量
    eig_value, eig_vector = numpy.linalg.eig(judge)
    max_eig = 0
    # 提取最大特征值。定理：正矩阵的最大特征值为正单根，对应特征向量为正向量
    for i in range(len(eig_value)):
        if eig_value[i].imag == 0 and eig_value[i].real > max_eig:
            max_eig = eig_value[i].real
            tmp = i
    # 在计算判断矩阵最大特征值后，原本应进行一致性校验来检验判断矩阵的合理性
    # 本程序中的judge判断矩阵已通过人工验算，满足一致性校验，因此在代码中略去该部分

    w = []
    w_sum = 0
    for i in range(len(eig_vector)):
        w.append(eig_vector[i][tmp].real)
        w_sum += eig_vector[i][tmp].real
    # 特征向量归一化得到权重向量
    w = [i / w_sum for i in w]
    return w


# 模糊综合评价表决器。函数参数为参与表决的执行体结果列表和ahp算法生成的权重向量
# 返回输出了表决结果的其中一个执行体的编号
def fce_voter(output_list, exe_selected, weight):
    L = len(exe_selected)
    flag = [0] * L
    score = []
    output = []
    k = 0

    # 对执行体输出的各个不同结果创建对象
    for i in range(L):
        # flag标志 防止重复生成对象
        if flag[i] == 0:
            # 创建输出结果对象
            output.append(OutputSet(exe_selected[i]))
            flag[i] = 1
            for j in range(i + 1, L, 1):
                if output_list[exe_selected[i]] == output_list[exe_selected[j]]:
                    # 记录输出了相同结果的执行体编号
                    flag[j] = 1
                    output[k].assentor_append(exe_selected[j])
            k += 1

    # 进行评分
    for i in range(k):
        score.append(output[i].evaluation(weight))
    """
    tmp = 0
    vote_result = None
    for i in range(k):
        if tmp < score[i]:
            tmp = score[i]
            vote_result = i
        elif tmp == score[i]:
            if output[vote_result].cs_read() < output[i].cs_read():
                tmp = score[i]
                vote_result = i
            # 评分与一致度均相同时不采纳
            elif output[vote_result].cs_read() == output[i].cs_read():
                vote_result = None"""

    # 评分最高的执行体结果作为表决结果
    vote_result = score.index(max(score))
    if vote_result is not None:
        i = 0
        # 读取输出了表决结果的执行体编号，进行各执行体采纳次数更新
        while k is not None:
            k = output[vote_result].assentor_read(i)
            if k is not None:
                adopted_time[k] += 1
            i += 1
        # 返回输出了表决结果的其中一个执行体的编号
        return output[vote_result].assentor_read(0)
    else:
        return None


# 一致表决，返回结果被采纳的执行体编号
def consensus_voter(output_list, exe_selected):
    # 用于存储各执行体响应结果的票数
    counter = [0] * exe_all
    # 防止重复计票
    flag = [0] * exe_all
    # 统计票数
    for x in range(0, exe_num, 1):
        if flag[exe_selected[x]] == 0:
            counter[exe_selected[x]] += 1
            flag[exe_selected[x]] == 1
            for y in range(x + 1, exe_num, 1):
                if output_list[exe_selected[x]] == output_list[exe_selected[y]]:
                    counter[exe_selected[x]] += 1
                    flag[exe_selected[y]] == 1
                # 若票数超过一半，则作为裁决结果，返回这个结果的索引编号
                if counter[exe_selected[x]] > exe_num/2:
                    return exe_selected[x]
    # 不存在票数多于一半的结果，则返回None
    # return None
    # 不存在票数多于一半的结果，则返回最大票数所对应的索引
    return counter.index(max(counter))
    if counter.count(max(counter)) == 1:
        return counter.index(max(counter))
    else:
        # 同时存在多个最大票数，返回None
        return None


# 多数表决，返回结果被采纳的执行体编号
def majority_voter(output_list, exe_selected):
    # 用于存储各执行体响应结果的票数
    counter = [0] * exe_all
    # 防止重复计票
    flag = [0] * exe_all
    # 统计票数
    for x in range(0, exe_num, 1):
        if flag[exe_selected[x]] == 0:
            counter[exe_selected[x]] += 1
            flag[exe_selected[x]] == 1
            for y in range(x + 1, exe_num, 1):
                if output_list[exe_selected[x]] == output_list[exe_selected[y]]:
                    counter[exe_selected[x]] += 1
                    flag[exe_selected[y]] == 1
                # 若票数超过一半，则作为裁决结果，返回这个结果的索引编号
                if counter[exe_selected[x]] > exe_num/2:
                    return exe_selected[x]
    # 不存在票数多于一半的结果，则返回None
    return None
    # 不存在票数多于一半的结果，则返回最大票数所对应的索引
    # return counter.index(max(counter))
    if counter.count(max(counter)) == 1:
        return counter.index(max(counter))
    else:
        # 同时存在多个最大票数，返回None
        return None


# 对仿真测试用的exe执行体计算异构度
# 执行体i和执行体j之间的异构度计算方式为：1-两个误差数据子集的杰卡德相似系数
# 如exe0=(1,...,1,2,2,3,3,3,3)，exe1=(1,...,1,2,3,3,3,4,5)，则对(2,2,3,3,3,3)和(2,3,3,3,4,5)求杰卡德相似系数
# 则exe0和exe1的异构度计算为：1-4/8
def test_he_matrix():
    for i in range(9):
        inter = union = 0
        for j in range(i+1, 10, 1):
            # 误差数据为2-5
            for k in range(2, 9, 1):
                # 求误差数据的交集、并集
                inter += min(exe[i].count(k), exe[j].count(k))
                union += max(exe[i].count(k), exe[j].count(k))
            he_all[j][i] = he_all[i][j] = round(1-inter/union, 2)


w = ahp()
exe_code = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
test_he_matrix()

rate = [0] * exe_all

cnt0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
cnt1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
cnt2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# 测试次数
test_time = 100000

tt = time.time()
# exe_selected = [0, 1, 2, 3, 4, 5, 6]
for t in range(test_time):
    exe_selected = random.sample(exe_code, exe_num)
    for i in exe_selected:
        on_time[i] += 1
        out[i] = random.choice(exe[i])

    re0 = fce_voter(out, exe_selected, w)
    re1 = consensus_voter(out, exe_selected)
    re2 = majority_voter(out, exe_selected)

    if re0 is not None:
        cnt0[out[re0]] += 1
    else:
        cnt0[0] += 1
    if re1 is not None:
        cnt1[out[re1]] += 1
    else:
        cnt1[0] += 1
    if re2 is not None:
        cnt2[out[re2]] += 1
    else:
        cnt2[0] += 1

    out = [None] * exe_all

print(cnt0, cnt1, cnt2)
# 置信度
for i in range(exe_all):
    rate[i] = round(adopted_time[i] / on_time[i], 6)
print('置信度：', rate)
print('A：', cnt0[1]*100/test_time, cnt1[1]*100/test_time, cnt2[1]*100/test_time)
# 执行体误差概率
for i in range(exe_all):
    rate[i] = 1-exe[i].count(1)/len(exe[i])
print('执行体误差概率：', rate)