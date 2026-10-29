"""中文/多语言交互预测：python predict.py --text '把 model 的参数调小一点'"""
import argparse
from langid_core import ROOT, LABELS, LABEL_NAMES, load_model, predict_proba, normalize

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='models/charcnn_aug_v2.pt',help='预训练模型文件')
    ap.add_argument('--text',default=None,help='直接预测一句话；省略进入交互模式')
    ap.add_argument('--topk',type=int,default=3)
    args=ap.parse_args()
    model,ckpt=load_model(ROOT/args.model)
    print(f'模型已加载：{ckpt["arch"]}，{len(LABELS)} 语种；输入 exit 退出')
    def one(txt):
        if not normalize(txt):
            print('没有可识别的文字（数字、标点和 emoji 会被过滤）');return
        pr=predict_proba(model,[txt],ckpt['vocab'],ckpt['max_len'])[0]
        top=pr.argsort()[::-1][:max(1,min(args.topk,len(LABELS)))]
        print('识别结果：'+' | '.join(f'{LABEL_NAMES[LABELS[i]]}({LABELS[i]}) {pr[i]:.1%}' for i in top))
        print('说明：百分比是 softmax 分数，不代表经过校准的可信概率。')
    if args.text is not None:
        one(args.text)
    else:
        while True:
            try: t=input('\n请输入一句话 > ').strip()
            except (KeyboardInterrupt,EOFError): break
            if t.lower() in ('exit','quit','q'):break
            one(t)
if __name__=='__main__':main()
