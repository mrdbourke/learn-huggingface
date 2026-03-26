# Work in Progress: Performing accelerated inference with vLLM

Note: This notebook is a work in progress.

Goal: 10x our inference speed on our LLM.

* TK - this notebook builds off of the LLM fine-tuning tutorial - https://www.learnhuggingface.com/notebooks/hugging_face_llm_full_fine_tune_tutorial 
* TK - reference link to our fine-tuned model
* TK - make a if/else statement to check if in Google Colab and then install required dependencies


```python
import time

print(f"Last updated: {time.ctime()}")
```


```python
import torch

assert torch.cuda.is_available()
```


```python
# =============================================================================
# vLLM vs HuggingFace Pipeline — Throughput Comparison (Google Colab)
# =============================================================================
# Run this entire cell in a Google Colab notebook with a GPU runtime.
# (Runtime > Change runtime type > T4 GPU or better)
#
# This demo:
#   1. Installs vllm
#   2. Loads the FoodExtract dataset
#   3. Runs inference with HuggingFace pipeline (baseline)
#   4. Runs inference with vLLM offline batch mode (fast)
#   5. Compares throughput and shows sample outputs
# =============================================================================

# --- Step 0: Install dependencies ---
# print("=" * 70)
# print("  STEP 0: Installing dependencies (this takes a few minutes)...")
# print("=" * 70)

# import subprocess, sys

# subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
#                        "vllm", "datasets", "matplotlib"])

# print("\n✅ Dependencies installed!\n")

# --- Step 1: Load dataset ---
print("=" * 70)
print("  STEP 1: Loading FoodExtract-1k dataset")
print("=" * 70)

from datasets import load_dataset

dataset = load_dataset("mrdbourke/FoodExtract-1k")

def sample_to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["sequence"]},
            {"role": "assistant", "content": sample["gpt-oss-120b-label-condensed"]},
        ]
    }

dataset = dataset.map(sample_to_conversation, batched=False)
print(f"✅ Loaded {len(dataset['train'])} samples\n")

# --- Step 2: Prepare prompts (using Gemma 3 chat template) ---
print("=" * 70)
print("  STEP 2: Preparing prompts")
print("=" * 70)

from transformers import AutoTokenizer

MODEL_PATH = "mrdbourke/FoodExtract-gemma-3-270m-fine-tune-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Format prompts using the chat template (same as the original notebook)
test_prompts = [
    tokenizer.apply_chat_template(
        item["messages"][:1],  # user message only
        tokenize=False,
        add_generation_prompt=True,
    )
    for item in dataset["train"]
]

print(f"✅ Prepared {len(test_prompts)} prompts")
print(f"   Example prompt (first 120 chars): {test_prompts[0][:120]}...\n")

# --- Step 3: HuggingFace Pipeline baseline ---
print("=" * 70)
print("  STEP 3: Running HuggingFace Pipeline (baseline)")
print("=" * 70)

import time
from transformers import AutoModelForCausalLM, pipeline, GenerationConfig

# Load model with HF
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="cuda",
    attn_implementation="eager",
)

hf_pipeline = pipeline(
    "text-generation",
    model=hf_model,
    tokenizer=tokenizer,
)

gen_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=False,
    disable_compile=True,
)

# Run HF pipeline with batch_size=32 (a reasonable batch size for comparison)
HF_BATCH_SIZE = 32
NUM_SAMPLES = len(test_prompts)

print(f"   Running {NUM_SAMPLES} samples with batch_size={HF_BATCH_SIZE}...")

hf_outputs = []
hf_start = time.time()

for i in range(0, NUM_SAMPLES, HF_BATCH_SIZE):
    batch = test_prompts[i : i + HF_BATCH_SIZE]
    batch_out = hf_pipeline(
        text_inputs=batch,
        batch_size=HF_BATCH_SIZE,
        generation_config=gen_config,
    )
    hf_outputs.extend(batch_out)
    # Progress update every 5 batches
    if ((i // HF_BATCH_SIZE) + 1) % 5 == 0:
        elapsed = time.time() - hf_start
        done = min(i + HF_BATCH_SIZE, NUM_SAMPLES)
        rate = done / elapsed
        print(f"   ... {done}/{NUM_SAMPLES} samples ({rate:.1f} samples/s)")

hf_time = time.time() - hf_start
hf_samples_per_sec = NUM_SAMPLES / hf_time

print(f"\n   ✅ HuggingFace Pipeline done!")
print(f"   Time: {hf_time:.2f}s | Samples/s: {hf_samples_per_sec:.2f}\n")

# Free HF model memory
del hf_model, hf_pipeline
import torch, gc
gc.collect()
torch.cuda.empty_cache()

# --- Step 4: vLLM offline batch inference ---
print("=" * 70)
print("  STEP 4: Running vLLM offline batch inference")
print("=" * 70)

from vllm import LLM, SamplingParams

# Load model with vLLM
# vLLM handles batching automatically — you just pass ALL prompts at once
vllm_model = LLM(
    model=MODEL_PATH,
    max_model_len=4096,       # keep it modest for Colab GPU memory
    gpu_memory_utilization=0.85,
    dtype="auto",
)

sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0,  # deterministic (greedy)
)

print(f"   Running {NUM_SAMPLES} samples in a single vLLM batch call...")

vllm_start = time.time()

# This is the key difference: ONE call with ALL prompts
# vLLM internally handles optimal batching, scheduling, and KV cache management
vllm_outputs = vllm_model.generate(test_prompts, sampling_params)

vllm_time = time.time() - vllm_start
vllm_samples_per_sec = NUM_SAMPLES / vllm_time

print(f"\n   ✅ vLLM batch inference done!")
print(f"   Time: {vllm_time:.2f}s | Samples/s: {vllm_samples_per_sec:.2f}\n")

# --- Step 5: Results comparison ---
print("=" * 70)
print("  RESULTS COMPARISON")
print("=" * 70)

speedup = hf_time / vllm_time

print(f"""
  {'Metric':<30} {'HuggingFace':>15} {'vLLM':>15}
  {'-' * 60}
  {'Total time (s)':<30} {hf_time:>15.2f} {vllm_time:>15.2f}
  {'Samples/second':<30} {hf_samples_per_sec:>15.2f} {vllm_samples_per_sec:>15.2f}
  {'Batch strategy':<30} {'chunk=' + str(HF_BATCH_SIZE):>15} {'auto (all)':>15}

  ⚡ vLLM speedup: {speedup:.1f}x faster

  At HF speed: {NUM_SAMPLES / hf_samples_per_sec / 3600:.2f} hours for {NUM_SAMPLES} samples
  At vLLM speed: {NUM_SAMPLES / vllm_samples_per_sec / 3600:.4f} hours for {NUM_SAMPLES} samples

  --- Extrapolation to 80M samples ---
  HF pipeline:  {80_000_000 / hf_samples_per_sec / 86400:,.1f} days
  vLLM batch:   {80_000_000 / vllm_samples_per_sec / 86400:,.1f} days
""")

# --- Step 6: Show sample outputs side by side ---
print("=" * 70)
print("  SAMPLE OUTPUTS (first 5)")
print("=" * 70)

for i in range(5):
    prompt_text = dataset["train"][i]["messages"][0]["content"]
    expected = dataset["train"][i]["messages"][1]["content"]

    # Extract HF output (strip the prompt prefix)
    hf_text = hf_outputs[i][0]["generated_text"][len(test_prompts[i]):]

    # Extract vLLM output
    vllm_text = vllm_outputs[i].outputs[0].text

    print(f"\n  --- Sample {i} ---")
    print(f"  Prompt:   {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
    print(f"  Expected: {expected[:80]}{'...' if len(expected) > 80 else ''}")
    print(f"  HF out:   {hf_text.strip()[:80]}{'...' if len(hf_text.strip()) > 80 else ''}")
    print(f"  vLLM out: {vllm_text.strip()[:80]}{'...' if len(vllm_text.strip()) > 80 else ''}")
    match = "✅ match" if hf_text.strip() == vllm_text.strip() else "⚠️  differ"
    print(f"  HF vs vLLM: {match}")

# --- Step 7: Plot ---
print(f"\n{'=' * 70}")
print("  PLOT")
print("=" * 70)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart: time
methods = ["HuggingFace\nPipeline", "vLLM\nBatch"]
times = [hf_time, vllm_time]
colors = ["steelblue", "coral"]

ax1.bar(methods, times, color=colors)
ax1.set_ylabel("Total Time (seconds)")
ax1.set_title(f"Inference Time — {NUM_SAMPLES} samples")
for i, v in enumerate(times):
    ax1.text(i, v + max(times) * 0.02, f"{v:.1f}s", ha="center", fontweight="bold")

# Add speedup arrow
ax1.annotate(
    f"{speedup:.1f}x faster",
    xy=(1, vllm_time + max(times) * 0.05),
    xytext=(0, hf_time + max(times) * 0.1),
    arrowprops=dict(arrowstyle="->", color="green", lw=2),
    fontsize=12, fontweight="bold", color="green", ha="center",
)
ax1.set_ylim(0, max(times) * 1.4)

# Bar chart: samples/s
rates = [hf_samples_per_sec, vllm_samples_per_sec]
ax2.bar(methods, rates, color=colors)
ax2.set_ylabel("Samples per Second")
ax2.set_title("Throughput Comparison")
for i, v in enumerate(rates):
    ax2.text(i, v + max(rates) * 0.02, f"{v:.1f}", ha="center", fontweight="bold")
ax2.set_ylim(0, max(rates) * 1.3)

plt.tight_layout()
plt.savefig("vllm_vs_hf_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Plot saved to vllm_vs_hf_comparison.png")
print("=" * 70)
print("  DONE!")
print("=" * 70)
```


```python

```


```python

```
