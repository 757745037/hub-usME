# make_vocab_v12.py 专门生成v12字符表
import os
from collections import Counter, OrderedDict
# 从训练脚本里导入需要的函数
from train_langid_cls_v12 import load_split, build_char_vocab, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
OUT_VOCAB = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")
def main():
    train_rows = load_split(DATA_DIR, "train")
    train_texts = [text for text, label in train_rows]
    # build_char_vocab内部会返回 vocab，但是freq是内部局部变量！
    vocab, freq = build_char_vocab(train_texts, min_freq=1)

    os.makedirs(os.path.dirname(OUT_VOCAB), exist_ok=True)
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            cnt = freq.get(ch, 0)
            f.write(f"{ch}\t{idx}\t{cnt}\n")

    print(f"字符表已生成完成！一共 {len(vocab)} 个符号")
    print(f"输出路径：{OUT_VOCAB}")

if __name__ == "__main__":
    main()
