# Work in progress: Batched LLM inference with Hugging Face Transformers

Note: This notebook is a work in progress.

* TK - this notebook builds off of the LLM fine-tuning tutorial - https://www.learnhuggingface.com/notebooks/hugging_face_llm_full_fine_tune_tutorial 


```python
import time

print(f"Last updated: {time.ctime()}")
```

## Bonus: Speeding up our model with batched inference

UPTOHERE TODO:

- Get the code working end to end
- Add intro explaining this requires knowledge from the previous notebook
- Break this notebook into sections
    - Goal
    - Intro
    - Step by step code to reproduce the workflow
    - Where to go next

--

Right now our model only inferences on one sample at a time but as is the case with many machine learning models, we could perform inference on multiple samples (also referred to as a batch) to significantly improve throughout.

In batched inference mode, your model performs predictions on X number of samples at once, this can dramatically improve sample throughput.

The number of samples you can predict on at once will depend on a few factors: 

* The size of your model (e.g. if your model is quite large, it may only be able to predict on 1 sample at time)
* The size of your compute VRAM (e.g. if your compute VRAM is already saturated, add multiple samples at a time may result in errors)
* The size of your samples (if one of your samples is 100x the size of others, this may cause errors with batched inference)

To find an optimal batch size for our setup, we can run an experiment:

* Loop through different batch sizes and measure the throughput for each batch size.
    * Why do we do this?
        * It's hard to tell the ideal batch size ahead of time.
        * So we experiment from say 1, 2, 4, 8, 16, 32, 64 batch sizes and see which performs best.
        * Just because we may get a speed up from using batch size 8, doesn't mean 64 will be better. 


```python
from datasets import load_dataset

dataset = load_dataset("mrdbourke/FoodExtract-1k")

print(f"[INFO] Number of samples in the dataset: {len(dataset['train'])}")

def sample_to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["sequence"]}, # Load the sequence from the dataset
            {"role": "system", "content": sample["gpt-oss-120b-label-condensed"]} # Load the gpt-oss-120b generated label
        ]
    }

# Map our sample_to_conversation function to dataset 
dataset = dataset.map(sample_to_conversation,
                      batched=False)

# Create a train/test split
dataset = dataset["train"].train_test_split(test_size=0.2,
                                            shuffle=False,
                                            seed=42)

# Number #1 rule in machine learning
# Always train on the train set and test on the test set
# This gives us an indication of how our model will perform in the real world
dataset
```

## TK - Load the model


```python
# Load the fine-tuned model and see how it goes
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "mrdbourke/FoodExtract-gemma-3-270m-fine-tune-v1"

print(f"[INFO] Loading in model from: {MODEL_ID}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
)

# Load trained model
loaded_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
);

# Check our loaded model (it's the same architecture as before except this time with updated weights)
loaded_model
```


```python
from transformers import pipeline

loaded_model_pipeline = pipeline("text-generation",
                                 model=loaded_model,
                                 tokenizer=tokenizer)

loaded_model_pipeline
```


```python
# Step 1: Need to turn our samples into batches (e.g. lists of samples)
print(f"[INFO] Formatting test samples into list prompts...")
test_input_prompts = [
    loaded_model_pipeline.tokenizer.apply_chat_template(
        item["messages"][:1],
        tokenize=False,
        add_generation_prompt=True
    )
    for item in dataset["test"]
]
print(f"[INFO] Number of test sample prompts: {len(test_input_prompts)}")
test_input_prompts[0]
```

## TK - Run batch predictions with manual batching


```python
# Step 2: Need to perform batched inference and time each step
import time
from tqdm.auto import tqdm

all_outputs = []

# Let's write a list of batch sizes to test
chunk_sizes_to_test = [1, 4, 8, 16, 32, 64, 128]
timing_dict = {}

# Loop through each batch size and time the inference
for CHUNK_SIZE in chunk_sizes_to_test:
    print(f"[INFO] Making predictions with batch size: {CHUNK_SIZE}")
    start_time = time.time()

    for chunk_number in tqdm(range(round(len(test_input_prompts) / CHUNK_SIZE))):
        batched_inputs = test_input_prompts[(CHUNK_SIZE * chunk_number): CHUNK_SIZE * (chunk_number + 1)]
        batched_outputs = loaded_model_pipeline(text_inputs=batched_inputs,
                                                batch_size=CHUNK_SIZE,
                                                max_new_tokens=256,
                                                disable_compile=True)
        
        all_outputs += batched_outputs
    
    end_time = time.time()
    total_time = end_time - start_time
    timing_dict[CHUNK_SIZE] = total_time
    print()
    print(f"[INFO] Total time for batch size {CHUNK_SIZE}: {total_time:.2f}s")
    print("="*80 + "\n\n")
```


```python
# TK - What does all_outputs do?
print(len(all_outputs))
```

## TK - Run batch predictions with `pipeline`'s automatic batching


```python
import time

chunk_sizes_to_test = [1, 4, 8, 16, 32, 64, 128]
timing_dict = {}

for batch_size in chunk_sizes_to_test:
    print(f"[INFO] Running with batch_size={batch_size}")
    start_time = time.time()

    outputs = loaded_model_pipeline(
        test_input_prompts,
        batch_size=batch_size,
        max_new_tokens=256,
    )

    elapsed = time.time() - start_time
    timing_dict[batch_size] = elapsed
    print(f"[INFO] batch_size={batch_size}: {elapsed:.2f}s | {len(test_input_prompts)/elapsed:.1f} samples/s\n")

print(timing_dict)
```


```python
batch_size
```

## TK - Run with KeyDataset


```python
# Run in the same script/notebook where loaded_model_pipeline, test_input_prompts, and ds are defined
import time
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

chunk_sizes_to_test = [1, 4, 8, 16, 32, 64, 128]
timing_dict = {}
all_outputs = {}

ds = Dataset.from_dict({"text": test_input_prompts})

for batch_size in chunk_sizes_to_test:
    print(f"[INFO] Running with batch_size={batch_size}")
    start_time = time.time()

    outputs = []
    for out in tqdm(
        loaded_model_pipeline(KeyDataset(ds, "text"), batch_size=batch_size, max_new_tokens=256),
        total=len(test_input_prompts),
        desc=f"batch_size={batch_size}",
    ):
        outputs.append(out)

    elapsed = time.time() - start_time
    timing_dict[batch_size] = elapsed
    all_outputs[batch_size] = outputs
    print(f"[INFO] batch_size={batch_size}: {elapsed:.2f}s | {len(test_input_prompts)/elapsed:.1f} samples/s\n")

# --- Verification: compare all batch sizes against batch_size=1 as baseline ---
baseline_size = chunk_sizes_to_test[0]
baseline_outputs = all_outputs[baseline_size]

print(f"{'Batch Size':<12} {'Length Match':<14} {'Outputs Match':<16} {'Time (s)':<10} {'Speedup vs bs=1'}")
print("=" * 70)

for batch_size in chunk_sizes_to_test:
    outputs = all_outputs[batch_size]
    len_match = len(outputs) == len(baseline_outputs)
    outputs_match = outputs == baseline_outputs
    speedup = timing_dict[baseline_size] / timing_dict[batch_size]

    print(f"{batch_size:<12} {str(len_match):<14} {str(outputs_match):<16} {timing_dict[batch_size]:<10.2f} {speedup:.2f}x")
```

## TK - Compare the results of batch (token size)


```python
# Run in the same script/notebook, after the benchmarking loop above
from difflib import SequenceMatcher

def extract_generated_text(output):
    """Extract the generated text string from a pipeline output.
    Adjust the key access depending on your pipeline's output format."""
    if isinstance(output, list):
        return output[0]["generated_text"]
    return output["generated_text"]

def token_similarity(text_a, text_b):
    """Simple token-level overlap: proportion of matching tokens."""
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    matcher = SequenceMatcher(None, tokens_a, tokens_b)
    return matcher.ratio()  # 0.0 to 1.0

# Compare all batch sizes against batch_size=1
baseline_size = chunk_sizes_to_test[0]
baseline_texts = [extract_generated_text(o) for o in all_outputs[baseline_size]]

print(f"{'Batch Size':<12} {'Length Match':<14} {'Avg Token Sim':<16} {'Time (s)':<10} {'Speedup vs bs=1'}")
print("=" * 70)

for batch_size in chunk_sizes_to_test:
    outputs = all_outputs[batch_size]
    candidate_texts = [extract_generated_text(o) for o in outputs]

    len_match = len(candidate_texts) == len(baseline_texts)

    similarities = [
        token_similarity(base, cand)
        for base, cand in zip(baseline_texts, candidate_texts)
    ]
    avg_sim = sum(similarities) / len(similarities) * 100

    speedup = timing_dict[baseline_size] / timing_dict[batch_size]

    sim_str = f"{avg_sim:.2f}%"
    print(f"{batch_size:<12} {str(len_match):<14} {sim_str:<16} {timing_dict[batch_size]:<10.2f} {speedup:.2f}x")
```

## TK - Compare the results of batching (speed wise)

Batched inference complete! Let's make a plot comparing different batch sizes.


```python
import matplotlib.pyplot as plt

# Data
data = timing_dict

total_samples = len(dataset["test"])

batch_sizes = list(data.keys())
inference_times = list(data.values())
samples_per_second = [total_samples / time for bs, time in data.items()]

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left plot: Total Inference Time ---
ax1.bar([str(bs) for bs in batch_sizes], inference_times, color='steelblue')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Total Inference Time (s)')
ax1.set_title('Inference Time by Batch Size')

for i, v in enumerate(inference_times):
    ax1.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)

# --- ARROW LOGIC (Left) ---
# 1. Identify Start (Slowest) and End (Fastest)
start_val = max(inference_times)
end_val = min(inference_times)
start_idx = inference_times.index(start_val)
end_idx = inference_times.index(end_val)

speedup = start_val / end_val

# 2. Draw Arrow (No Text)
# connectionstyle "rad=-0.3" arcs the arrow upwards
ax1.annotate("",
             xy=(end_idx, end_val+(0.5*end_val)),
             xytext=(start_idx+0.25, start_val+10),
             arrowprops=dict(arrowstyle="->", color='green', lw=1.5, connectionstyle="arc3,rad=-0.3"))

# 3. Place Text at Midpoint
mid_x = (start_idx + end_idx) / 2
# Place text slightly above the highest point of the two bars
text_y = max(start_val, end_val) + (max(inference_times) * 0.1)

ax1.text(mid_x+0.5, text_y-150, f"{speedup:.1f}x speedup",
         ha='center', va='bottom', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

ax1.set_ylim(0, max(inference_times) * 1.35) # Increase headroom for text


# --- Right plot: Samples per Second ---
ax2.bar([str(bs) for bs in batch_sizes], samples_per_second, color='coral')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Samples per Second')
ax2.set_title('Throughput by Batch Size')

for i, v in enumerate(samples_per_second):
    ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)

# --- ARROW LOGIC (Right) ---
# 1. Identify Start (Slowest) and End (Fastest)
start_val_t = min(samples_per_second)
end_val_t = max(samples_per_second)
start_idx_t = samples_per_second.index(start_val_t)
end_idx_t = samples_per_second.index(end_val_t)

speedup_t = end_val_t / start_val_t

# 2. Draw Arrow (No Text)
ax2.annotate("",
             xy=(end_idx_t-(0.05*end_idx_t), end_val_t+(0.025*end_val_t)),
             xytext=(start_idx_t, start_val_t+0.6),
             arrowprops=dict(arrowstyle="->", color='green', lw=1.5, connectionstyle="arc3,rad=-0.3"))

# 3. Place Text at Midpoint
mid_x_t = (start_idx_t + end_idx_t) / 2
text_y_t = max(start_val_t, end_val_t) + (max(samples_per_second) * 0.1)

ax2.text(mid_x_t-0.5, text_y_t-4.5, f"{speedup_t:.1f}x speedup",
         ha='center', va='bottom', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

ax2.set_ylim(0, max(samples_per_second) * 1.35) # Increase headroom

plt.suptitle("Inference with Fine-Tuned Gemma 3 270M on NVIDIA DGX Spark")
plt.tight_layout()
plt.savefig('inference_benchmark.png', dpi=150)
plt.show()
```

We get a 4-6x speedup when using batches!

At 10 samples per second, that means we can inference on ~800k samples in a day.


```python
samples_per_second = round(len(dataset["test"]) / min(timing_dict.values()), 2)
seconds_in_a_day = 86_400
samples_per_day = seconds_in_a_day * samples_per_second

print(f"[INFO] Number of samples per second: {samples_per_second} | Number of samples per day: {samples_per_day}")
```

## TK - Extensions

TK - next: vllm inference for faster modelling


```python

```


```python

```


```python

```


```python

```
