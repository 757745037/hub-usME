"""训练入口。例：python train.py --arch cnn --augment --epochs 24 --output models/charcnn_aug.pt"""
import argparse
import csv
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from langid_core import ROOT, read_rows, build_vocab, LanguageDataset, create_model, seed_all, predict_proba, metrics_from_prob, save_model, load_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arch',choices=['rnn','cnn'],default='cnn')
    ap.add_argument('--augment',action='store_true')
    ap.add_argument('--epochs',type=int,default=24)
    ap.add_argument('--patience',type=int,default=7)
    ap.add_argument('--batch-size',type=int,default=96)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--max-len',type=int,default=80)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output',default='models/retrained.pt')
    ap.add_argument('--resume',default=None,help='从现有模型权重继续优化')
    ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'],help='优先使用 CUDA GPU')
    args=ap.parse_args()
    seed_all(args.seed);torch.set_num_threads(1)
    train=read_rows(ROOT/'data/train.csv');dev=read_rows(ROOT/'data/dev.csv')
    vocab, counts=build_vocab(train)
    vocab_path=ROOT/'vocab/vocab_v12.txt'
    vocab_path.parent.mkdir(parents=True,exist_ok=True)
    with open(vocab_path,'w',encoding='utf-8') as f:
        f.write('# character\tid\ttrain_frequency\n<PAD>\t0\t0\n<UNK>\t1\t0\n')
        for ch,idx in list(vocab.items())[2:]:f.write(f'{ch}\t{idx}\t{counts[ch]}\n')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    model=create_model(args.arch,len(vocab))
    if args.resume:
        old, ck=load_model(ROOT/args.resume)
        assert ck['arch']==args.arch and ck['vocab']==dict(vocab) and ck['max_len']==args.max_len
        model.load_state_dict(old.state_dict())
    model=model.to(device)
    loader=DataLoader(LanguageDataset(train,vocab,args.max_len,args.augment,args.seed),batch_size=args.batch_size,shuffle=True,num_workers=0)
    devtxt=[r['text'] for r in dev]
    loss_fn=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3)
    path=ROOT/args.output; path.parent.mkdir(parents=True,exist_ok=True)
    best=-1;bad=0;history=[]; start=time.perf_counter()
    print('device:',device,'arch:',args.arch,'augmentation:',args.augment,'train/dev:',len(train),len(dev),'vocab:',len(vocab),'params:',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();total=0;seen=0
        for X,y in loader:
            X,y=X.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(X);loss=loss_fn(logits,y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();total+=loss.item()*len(y);seen+=len(y)
        probs=predict_proba(model,devtxt,vocab,args.max_len)
        m=metrics_from_prob(probs,dev)
        sched.step(m['macro_f1'])
        history.append(dict(epoch=epoch,loss=round(total/seen,6),dev_accuracy=m['accuracy'],dev_macro_f1=m['macro_f1'],lr=opt.param_groups[0]['lr']))
        print(f"{epoch:02d} loss={total/seen:.4f} dev_acc={m['accuracy']:.4f} dev_f1={m['macro_f1']:.4f}",flush=True)
        if m['macro_f1']>best+1e-5:
            best=m['macro_f1'];bad=0;save_model(path,model,args.arch,vocab,args.max_len,epoch,best)
        else:bad+=1
        if bad>=args.patience:
            print('early stop (dev macro-F1)',flush=True);break
    hist=path.with_suffix('.history.json')
    hist.write_text(json.dumps(dict(config=vars(args),best_dev_macro_f1=best,history=history,seconds=round(time.perf_counter()-start,2)),ensure_ascii=False,indent=2),encoding='utf8')
    print('saved',path,'best dev F1:',best,'seconds:',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
