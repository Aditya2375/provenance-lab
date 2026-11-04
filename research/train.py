"""Small, grouped Wikipedia provenance experiment. Not an authorship detector."""
import hashlib,json,random,re,time,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
SEED=37;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs'/'model';OUT.mkdir(parents=True,exist_ok=True)
base='google/bert_uncased_L-2_H-128_A-2'
DATA=Path(os.environ.get('DATA_DIR','/tmp/textdata'))
rows=[json.loads(x) for x in open(DATA/'paras_en_first_gpt4o.jsonl')];other={x['id']:x for x in map(json.loads,open(DATA/'paras_en_first_mistral.jsonl'))}
def split(title):
 n=int(hashlib.sha256(('provenance37:'+title).encode()).hexdigest()[:8],16)%100
 return 'train' if n<70 else 'validation' if n<85 else 'test'
def clean(s):return re.sub(r'\s+',' ',s).strip()
data={k:[] for k in ['train','validation','test']};seen={}
# Group by page title; filter texts shared across titles/splits before training.
for r in rows:
 k=split(r['page_title'])
 for label,text in [(0,r['trgt']),(1,r['mgt'])]:
  text=clean(text)
  if len(text.split())<40:continue
  h=hashlib.sha256(text.encode()).hexdigest();seen.setdefault(h,set()).add(k)
  data[k].append(dict(text=text,label=label,id=r['id'],page=r['page_title'],hash=h))
for k in data:
 unique={}
 for x in data[k]:
  if len(seen[x['hash']])==1:unique[x['hash']]=x
 data[k]=list(unique.values())
print('splits',[(k,len(v)) for k,v in data.items()],flush=True)
tokenizer=AutoTokenizer.from_pretrained(base,revision='30b0a37ccaaa32f332884b96992754e246e48c5f')
model=AutoModelForSequenceClassification.from_pretrained(base,revision='30b0a37ccaaa32f332884b96992754e246e48c5f',num_labels=2)
# Exact model revision stored for reproduction.
revision='30b0a37ccaaa32f332884b96992754e246e48c5f'
enc={k:tokenizer([x['text'] for x in v],padding='max_length',truncation=True,max_length=128,return_tensors='pt') for k,v in data.items()}
labels={k:torch.tensor([x['label'] for x in v]) for k,v in data.items()}
def loader(k,shuffle=False):return DataLoader(TensorDataset(enc[k]['input_ids'],enc[k]['attention_mask'],enc[k]['token_type_ids'],labels[k]),batch_size=32,shuffle=shuffle,generator=torch.Generator().manual_seed(SEED))
def probs(k):
 model.eval();out=[]
 with torch.no_grad():
  for ids,mask,types,y in loader(k):out.extend(model(input_ids=ids,attention_mask=mask,token_type_ids=types).logits.softmax(-1)[:,1].tolist())
 return np.array(out)
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01);best=-1;history=[]
for epoch in range(0 if os.environ.get('EVAL_ONLY') else 3):
 model.train();losses=[];start=time.time()
 for step,(ids,mask,types,y) in enumerate(loader('train',True)):
  optimizer.zero_grad();out=model(input_ids=ids,attention_mask=mask,token_type_ids=types,labels=y);out.loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(out.loss.item())
 val=probs('validation');score=roc_auc_score(labels['validation'],val);history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_auc=score,seconds=time.time()-start));print(history[-1],flush=True)
 if score>best:best=score;torch.save(model.state_dict(),'/tmp/provenance-best.pt')
if os.environ.get('EVAL_ONLY'): history=json.load(open(ROOT/'docs'/'metrics.json'))['history']
model.load_state_dict(torch.load('/tmp/provenance-best.pt',weights_only=True));val=probs('validation');test=probs('test')
# Threshold selected solely on validation accuracy, tie closest to0.5.
threshold=max(np.linspace(.1,.9,81),key=lambda t:(accuracy_score(labels['validation'],val>=t),-abs(t-.5)))
def metric(y,p,t=None):
 t=threshold if t is None else t
 cm=confusion_matrix(y,p>=t,labels=[0,1]);tn,fp,fn,tp=cm.ravel();return dict(n=len(y),accuracy=accuracy_score(y,p>=t),auc=roc_auc_score(y,p),f1=f1_score(y,p>=t),confusion=cm.tolist(),false_positive_rate=fp/(fp+tn),false_negative_rate=fn/(fn+tp))
# Held-out Mistral examples from only test-page ids; human comparison same test human rows.
ood=[]
for x in data['test']:
 if x['label']==0:ood.append(x)
 elif x['id'] in other:ood.append(dict(text=clean(other[x['id']]['mgt']),label=1,id=x['id'],page=x['page']))
model.eval()
with torch.no_grad():
 oodp=[]
 for i in range(0,len(ood),32):
  tokens=tokenizer([x['text'] for x in ood[i:i+32]],padding='max_length',truncation=True,max_length=128,return_tensors='pt');oodp.extend(model(**tokens).logits.softmax(-1)[:,1].tolist())
# Lexical baseline to reveal how much artifact separation explains.
vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=15000,sublinear_tf=True);X=vec.fit_transform(x['text'] for x in data['train']);clf=LogisticRegression(max_iter=500,random_state=SEED);clf.fit(X,labels['train']);bp=clf.predict_proba(vec.transform(x['text'] for x in data['test']))[:,1]
bv=clf.predict_proba(vec.transform(x['text'] for x in data['validation']))[:,1]
bt=max(np.linspace(.1,.9,81),key=lambda t:(accuracy_score(labels['validation'],bv>=t),-abs(t-.5)))
metrics=dict(baseline_threshold=float(bt),seed=SEED,base_model=base,base_revision=revision,dataset='GerritQ/tsm_bench',dataset_revision='3f9cdae5a4a4715f14133872fd5db16ebfead474',files=['paras/paras_en_first_gpt4o.jsonl','paras/paras_en_first_mistral.jsonl'],max_tokens=128,epochs=3,batch_size=32,learning_rate=5e-5,threshold=float(threshold),history=history,splits={k:dict(examples=len(v),pages=len(set(x['page'] for x in v)),class_counts=[sum(x['label']==i for x in v) for i in [0,1]]) for k,v in data.items()},validation=metric(labels['validation'],val),test=metric(labels['test'],test),mistral_heldout=metric([x['label'] for x in ood],np.array(oodp)),lexical_baseline=metric(labels['test'],bp,bt))
json.dump(metrics,open(ROOT/'docs'/'metrics.json','w'),indent=2);json.dump({k:[{'id':x['id'],'page':x['page'],'label':x['label'],'sha256':x['hash']} for x in v] for k,v in data.items()},open(ROOT/'research'/'split_manifest.json','w'),indent=2)
model.save_pretrained('/tmp/provenance-trained');tokenizer.save_pretrained(OUT)
class Wrapped(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,input_ids,attention_mask,token_type_ids):return self.m(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).logits
example=tokenizer('A short Wikipedia paragraph.',padding='max_length',max_length=128,return_tensors='pt');wrapped=Wrapped(model).eval()
torch.onnx.export(wrapped,tuple(example[k] for k in ['input_ids','attention_mask','token_type_ids']),str(OUT/'model.onnx'),input_names=['input_ids','attention_mask','token_type_ids'],output_names=['logits'],dynamic_axes={k:{0:'batch',1:'sequence'} for k in ['input_ids','attention_mask','token_type_ids']},opset_version=17,dynamo=False)
from onnxruntime.quantization import quantize_dynamic,QuantType
quantize_dynamic(str(OUT/'model.onnx'),str(OUT/'2-model-int8.onnx'),weight_type=QuantType.QInt8)
print(json.dumps(metrics,indent=2),flush=True)

# The public manifest is sharded for browser-editor transport limits.
for k,v in data.items():
 rows=[dict(id=x["id"],page=x["page"],label=x["label"],sha256=x["hash"]) for x in v]
 for i in range(0,len(rows),700):
  json.dump(rows[i:i+700],open(ROOT/"research"/("manifest-"+k+"-"+str(i//700)+".json"),"w"),ensure_ascii=False,separators=(",",":"))
