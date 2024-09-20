# Qwen-VL-Chat 

## Repository Structure
```
# Data resource for training 
data
├── nikluge-gips-2023-train.jsonl       # Training data
├── nikluge-gips-2023-test.jsonl        # Test data
├── nikluge-gips-2023-dev.jsonl         # Development data
├── nikluge-gips-2023-train-qwen.jsonl  # Qwen-specific training data
├── nikluge-gips-2023-test-qwen.jsonl   # Qwen-specific test data
├── nikluge-gips-2023-dev-qwen.jsonl    # Qwen-specific development data
└── convert.py                          # Script to convert data to Qwen format

# Code for model fine-tuning and inference
finetune
├── run_inference.sh          # Script to run inference
├── finetune_qlora_ds.sh      # Script for QLoRA fine-tuning
├── finetune_lora_ds.sh       # Script for LoRA fine-tuning
├── finetune_ds.sh            # Script for DeepSpeed fine-tuning
├── ds_config_zero3.json      # DeepSpeed ZeRO-3 configuration
└── ds_config_zero2.json      # DeepSpeed ZeRO-2 configuration

# Inference results
inference
├── qwen_results_V2.jsonl     # Version 2 of inference results
└── qwen_results_V1.jsonl     # Version 1 of inference results

# Evaluation scripts and data
eval_mm
├── vqa_eval.py
├── seed_bench
├── mmbench
├── evaluate_multiple_choice.py
├── evaluate_caption.py
├── EVALUATION.md
├── vqa.py
├── mme
├── evaluate_vqa.py
├── evaluate_grounding.py
└── data

# Main scripts
inference.py                  # Script for running inference
finetune.py                   # Script for fine-tuning
openai_api.py                 # OpenAI-like API implementation

# Additional files
requirements.txt              # Python package requirements
SimSun.ttf                    # Font file
README.md                     # This file

# Output directory for fine-tuned models
output_qwen
├── checkpoint-02
└── checkpoint-01

# Hugging Face cache
hf_cache
├── modules
└── hub

# Test environment
qwen_test
├── share
├── pyvenv.cfg
├── lib64
├── lib
├── include
└── bin

# Asset files
assets
├── Various image files...
```

## Data
### How to Prepare Data
```
  {
    "id": "identity_train_07278",
    "conversations": [
      {
        "from": "user",
        "value": "<img>/data/swbaek/Projects/Korean_IC_Competition/data/nikluge-gips-2023_image/PK2123.jpg</img>\n 효교로\n 이미지를 설명해주세요."
      },
      {
        "from": "assistant",
        "value": "회색 버스 정류장 기둥에는 효교로 52가 적힌 파란색 스티커가 붙어 있다."
      },
      {
        "from": "assistant",
        "value": "효교로 52임을 표시하는 파란 스티커가 회색 바탕의 버스 정류장 기둥에 부착되어 있다."
      },
      {
        "from": "assistant",
        "value": "효교로 52라고 적혀 있는 파란색 스티커가 부착된 위치는 회색 버스 정류장 기둥의 겉면이다."
      },
      {
        "from": "assistant",
        "value": "파란색 스티커가 부착된 곳은 버스 정류장의 회색 기둥이고, 스티커에는 효교로 52가라고 쓰여 있다."
      },
      {
        "from": "assistant",
        "value": "회색 버스 정류장 기둥에 붙은 파란 스티커에 효교로 52라고 적혀 있다."
      }
    ]
  },
...
...
```

## Installation
Prerequisites
- Ensure you have Python 3.10 or higher installed.

Setting Up the Environment
1. Clone the Repository (if applicable):

```
git clone <repository-url>
cd Qwen-VL-fine-tuning
```

2. Install Dependencies:
```
pip install -r requirements.txt
```

## How to Run (Training, Inference, Scoring)
# Note: Please change your configurations through modifying the relevant script files in the finetune folder

### Fine-tuning
```
chmod +x finetune/finetune_ds.sh
./finetune/finetune_ds.sh
```
- Fine-tuning logs and models will be saved in the output_qwen directory.

### Inference
```
chmod +x inference/run_inference.sh
./inference/run_inference.sh
```
- During inference, the output data is saved in jsonl format in the inference directory.

### Evaluation

- Evaluation scripts are provided in the eval_mm directory. Refer to EVALUATION.md for detailed information on how to run evaluations.

## License
MIT License

## Credits
This code was developed by the Sionic AI research team. For any inquiries or further information regarding this code, please reach out to Sangwon Baek at baeksw98@sionic.ai.
To learn more about our company and our vision for the future of AI, please visit our website at https://sionic.ai/.