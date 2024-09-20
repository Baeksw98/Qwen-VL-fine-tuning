import argparse
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main(args):
    print(f"[+] Using device: {args.device}")
    device = torch.device(args.device)

    print(f"[+] Loading model and tokenizer from {args.model_path}")
    try:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            device_map='auto',
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print("[+] Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"[-] Error loading model or tokenizer: {str(e)}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)

    print(f"[+] Opening input file: {args.data_path}")
    if not os.path.exists(args.data_path):
        print(f"[-] Input file not found: {args.data_path}")
        return

    print(f"[+] Image directory: {args.image_dir}")
    if not os.path.exists(args.image_dir):
        print(f"[-] Image directory not found: {args.image_dir}")
        return

    print("[+] Start Inference")
    with open(args.data_path, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        print("[+] Files opened successfully")
        for line in tqdm(f_in):
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"[-] Error decoding JSON: {line[:50]}...")
                continue

            img_path = os.path.join(args.image_dir, item['input']['id'] + '.jpg')
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                continue

            # Extract OCR text
            ocr_text = ' '.join([ocr['words'] for ocr in item['input']['ocr_info']])

            try:
                if hasattr(tokenizer, 'from_list_format'):
                    query = tokenizer.from_list_format([
                        {'image': img_path},
                        {'text': f'이미지에 대한 설명을 해주세요. OCR 텍스트: {ocr_text}'},
                    ])
                else:
                    # Fallback for older versions or different tokenizers
                    query = tokenizer(f'이미지에 대한 설명을 해주세요. OCR 텍스트: {ocr_text}', return_tensors="pt").to(device)
                    
                if hasattr(model, 'chat'):
                    response, _ = model.chat(tokenizer, query=query, history=None)
                else:
                    # Fallback for models without chat method
                    outputs = model.generate(**query, max_new_tokens=100)
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Create the output in the desired format
                output_item = {
                    "id": item["id"],
                    "input": item["input"],
                    "output": response
                }
                
                json.dump(output_item, f_out, ensure_ascii=False)
                f_out.write('\n')
            except Exception as e:
                print(f"Error processing {item['id']}: {str(e)}")

    print(f"[+] Inference complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    main(args)