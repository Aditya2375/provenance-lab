"""Fetch only the two named files at the recorded dataset revision."""
from pathlib import Path
from urllib.request import urlopen
import os
p=Path(os.environ.get('DATA_DIR','/tmp/textdata'));p.mkdir(parents=True,exist_ok=True)
rev='3f9cdae5a4a4715f14133872fd5db16ebfead474'
for name in ['paras_en_first_gpt4o.jsonl','paras_en_first_mistral.jsonl']:
 url='https://huggingface.co/datasets/GerritQ/tsm_bench/resolve/'+rev+'/paras/'+name
 # URL is assembled solely from the fixed HTTPS host/revision and two literal filenames above.
 # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
 with urlopen(url,timeout=60) as response:
  (p/name).write_bytes(response.read())
 print('Saved',name)
