#encoding=utf-8
import math
import jieba,sys
import jieba.analyse
reload(sys)
sys.setdefaultencoding('utf8')
class SimHash(object):

    def __init__(self):
        pass

    def getBinStr(self, source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            print(source, x)

            return str(x)


    def getWeight(self, source):
        # fake weight with keyword
        return ord(source)
    def unwrap_weight(self, arr):
        ret = ""
        for item in arr:
            tmp = 0
            if int(item) > 0:
                tmp = 1
            ret += str(tmp)
        return ret

    def simHash(self, rawstr):
        seg = jieba.cut(rawstr, cut_all=True)
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=100, withWeight=True)
        print(keywords)
        ret = []
        for keyword, weight in keywords:
            binstr = self.getBinStr(keyword)
            keylist = []
            for c in binstr:
                weight = math.ceil(weight)
                if c == "1":
                    keylist.append(int(weight))
                else:
                    keylist.append(-int(weight))
            ret.append(keylist)
        # 对列表进行"降维"
        rows = len(ret)
        cols = len(ret[0])
        result = []
        for i in range(cols):
            tmp = 0
            for j in range(rows):
                tmp += int(ret[j][i])
            if tmp > 0:
                tmp = "1"
            elif tmp <= 0:
                tmp = "0"
            result.append(tmp)
        return "".join(result)

    def getDistince(self, hashstr1, hashstr2):
        length = 0
        for index, char in enumerate(hashstr1):
            if char == hashstr2[index]:
                continue
            else:
                length += 1
        return length


if __name__ == "__main__":
    simhash = SimHash()
    s1="SimHash是一种局部敏感hash，它也是Google公司进行海量网页去重使用的主要算法。传统的Hash算法只负责将原始内容尽量均匀随机地映射为一个签名值，原理上仅相当于伪随机数产生算法。传统的hash算法产生的两个签名，如果原始内容在一定概率下是相等的；如果不相等，除了说明原始内容不相等外，不再提供任何信息，因为即使原始内容只相差一个字节，所产生的签名也很可能差别很大。所以传统的Hash是无法在签名的维度上来衡量原内容的相似度，而SimHash本身属于一种局部敏感哈希算法，它产生的hash签名在一定程度上可以表征原内容的相似度。我们主要解决的是文本相似度计算，要比较的是两个文章是否相似，当然我们降维生成了hash签名也用于这个目的。看到这里估计大家就明白了，我们使用的simhash就算把文章中的字符串变成 01 串也还是可以用于计算相似度的，而传统的hash却不行。"
    s2="simhash是一个本地敏感的哈希，也是Google用来重用海量网页的主要算法。传统的散列算法只负责将原始内容尽可能均匀、随机地映射到签名值，原则上只相当于伪随机数生成算法。传统的散列算法产生两个签名，如果原始内容在一定概率下是相等的；如果不相等，除了原始内容不相等之外，它不会提供任何信息，因为即使原始内容相距只有一个字节，生成的签名也可能非常不同。因此，传统的散列算法无法在签名维数上度量原始内容的相似性。simhash本身属于本地敏感的哈希算法。simhash生成的散列签名在一定程度上可以表示原始内容的相似性。我们主要解决文本相似度计算的问题，为了比较两篇文章是否相似，当然，为此我们通过降维生成散列签名。从中可以看到，即使将文章中的字符串转换为01字符串，simhash也可以用来计算相似性，但是传统的hash不能。"
    with open("a.txt", "r") as file:
        s1 = "".join(file.readlines())
        file.close()
    with open("b.txt", "r") as file:
        s2 = "".join(file.readlines())
        file.close()
    # s1 = "this is just test for simhash, here is the difference"
    # s2 = "this is a test for simhash, here is the difference"
    # print(simhash.getBinStr(s1))
    # print(simhash.getBinStr(s2))
    hash1 = simhash.simHash(s1)
    hash2 = simhash.simHash(s2)
    distince = simhash.getDistince(hash1, hash2)
    # value = math.sqrt(len(s1)**2 + len(s2)**2)
    value = 5
    print("海明距离：", distince, "判定距离：", value, "是否相似：", distince<=value)
