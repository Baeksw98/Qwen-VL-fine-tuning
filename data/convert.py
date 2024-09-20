import json
import os
import glob


def convert_jsonl_to_conversation_format(input_file, output_file, image_dir, dataset_type, is_test=False, start_id=1):
    conversations = []
    # Build a mapping from image ID to image file path
    image_files = {
        os.path.splitext(os.path.basename(f))[0].strip().lower(): f
        for f in glob.glob(os.path.join(image_dir, '*.jpg'))
    }
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f, start=start_id):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_id}: {e}")
                continue
            
            # Validate required fields
            if 'input' not in data or 'id' not in data['input']:
                print(f"Missing 'input' or 'id' in data on line {line_id}. Skipping.")
                continue
            if not is_test and 'output' not in data:
                print(f"Missing 'output' in data on line {line_id}. Skipping.")
                continue
            
            image_id = data['input']['id'].strip().lower()
            if image_id not in image_files:
                print(f"Warning: Image file for {image_id} not found in {dataset_type} dataset.")
                continue
            
            image_path = image_files[image_id]
            
            # Validate OCR info
            ocr_info = data['input'].get('ocr_info', [])
            if not ocr_info or 'words' not in ocr_info[0]:
                print(f"Missing 'ocr_info' or 'words' in data on line {line_id}. Skipping.")
                continue
            
            user_input = f"<img>{image_path}</img>\n {ocr_info[0]['words']}\n 이미지를 설명해주세요."
            
            if is_test:
                # For test data, only include the user input
                conversation = {
                    "id": f"identity_{dataset_type}_{line_id:05d}",
                    "conversations": [
                        {
                            "from": "user",
                            "value": user_input
                        }
                    ]
                }
                conversations.append(conversation)
            else:
                # For train and dev data, create separate entries for each assistant response
                outputs = data.get('output', [])
                if not outputs:
                    print(f"Warning: No outputs for {image_id} in {dataset_type} dataset.")
                    continue
                for assistant_idx, output in enumerate(outputs, start=1):
                    conversation_id = f"identity_{dataset_type}_{line_id:05d}_{assistant_idx:04d}"
                    conversation = {
                        "id": conversation_id,
                        "conversations": [
                            {
                                "from": "user",
                                "value": user_input
                            },
                            {
                                "from": "assistant",
                                "value": output
                            }
                        ]
                    }
                    conversations.append(conversation)
                    
    # Write the result to the output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(conversations, f_out, ensure_ascii=False, indent=2)
    
    # Return the number of processed entries
    return len(conversations), line_id

# Directory paths
base_dir = '/data/swbaek/Projects/Korean_IC_Competition/data'
image_dir = os.path.join(base_dir, 'nikluge-gips-2023_image')
input_json_dir = os.path.join(base_dir, 'nikluge-gips-2023_JSONL')
output_json_dir = '/data/swbaek/Projects/Korean_IC_Competition/Qwen-VL-fine-tuning/data'

# Ensure the output directory exists
os.makedirs(output_json_dir, exist_ok=True)

# Process train, dev, and test data
datasets = [
    ('nikluge-gips-2023-train.jsonl', 'nikluge-gips-2023-train-qwen.jsonl', 'train', False),
    ('nikluge-gips-2023-dev.jsonl', 'nikluge-gips-2023-dev-qwen.jsonl', 'dev', False),
    ('nikluge-gips-2023-test.jsonl', 'nikluge-gips-2023-test-qwen.jsonl', 'test', True)
]

total_processed = 0
current_id = 0

for input_file, output_file, dataset_type, is_test in datasets:
    input_path = os.path.join(input_json_dir, input_file)
    output_path = os.path.join(output_json_dir, output_file)
    
    processed, last_id = convert_jsonl_to_conversation_format(
        input_path, output_path, image_dir, dataset_type, is_test, current_id
    )
    total_processed += processed
    current_id = last_id + 1
    print(f"Conversion complete for {input_file}. Output saved to {output_file}. Processed {processed} entries.")

print(f"Total number of entries processed: {total_processed}")