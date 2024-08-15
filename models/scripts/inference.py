from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


instruction = 'FINDINGS:The lungs are clear of consolidation. Linear left basilar opacity is most likely atelectasis versus scarring. The cardiomediastinal silhouette is within normal limits. Median sternotomy wires are again noted. There is no free air below the diaphragm.IMPRESSION:No acute cardiopulmonary process. No free intraperitoneal air.\nBased on the above information, answer the question.\nQuestion: Please provide detailed and comprehensive diagnostic results.'
model_path = 'MEDSQ'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
response = model.chat(tokenizer, instruction, history=None, eos_token_id=2, pad_token_id=2, temperature=0.3, top_p=0.8, max_length=None, max_new_tokens=512)[0]
print(response)
