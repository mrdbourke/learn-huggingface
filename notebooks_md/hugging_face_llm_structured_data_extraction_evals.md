# Work in progress: Batched LLM inference

Note: This notebook is a work in progress.

Goal: Batched inference with Hugging Face `transformers`.

TK - Resources
* TK - this notebook builds off of the LLM fine-tuning tutorial - https://www.learnhuggingface.com/notebooks/hugging_face_llm_full_fine_tune_tutorial 


```python
import time

print(f"Last updated: {time.ctime()}")
```



* TK - use JSON here to make sure our model outputs the right output
* Code:

```python
import json

def validate_response(response_text: str, required_fields: dict) -> tuple[bool, str]:
    """
    required_fields = {"answer": str, "confidence": float}
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    for field, expected_type in required_fields.items():
        if field not in data:
            return False, f"Missing field: {field}"
        if not isinstance(data[field], expected_type):
            return False, f"{field} expected {expected_type.__name__}, got {type(data[field]).__name__}"

    return True, "valid"

# Usage
raw = '{"answer": "Paris", "confidence": 0.95}'
ok, msg = validate_response(raw, {"answer": str, "confidence": float})
```

* dataclass?

```
# models.py
from dataclasses import dataclass, field, asdict
from typing import TypedDict

# --- dataclass: mutable objects you build up during a run ---
@dataclass
class EvalResult:
    prompt: str
    expected: str
    actual: str = ""
    latency_s: float = 0.0
    passed: bool = False
    error: str | None = None

# Create, mutate, then serialize
result = EvalResult(prompt="Capital of France?", expected="Paris")
result.actual = "Paris"
result.passed = result.actual.lower() == result.expected.lower()
result.latency_s = 0.432

print(asdict(result))  # clean dict for JSON output


# --- TypedDict: typing for dicts you're reading (e.g. from a JSON file) ---
class TestCase(TypedDict):
    id: int
    prompt: str
    expected_answer: str

# This is just a type hint — it's still a plain dict at runtime
case: TestCase = {"id": 1, "prompt": "Capital of France?", "expected_answer": "Paris"}
```



```python

```

## TK - Bonus: Performing evaluations on our model

TK - split this into another notebook, introduce batching to make the predictions faster

We can evaluate our model directly against the original labels from `gpt-oss-120b` and see how it stacks up.

For example, we could compare the following:

* F1 Score
* Precision
* Recall
* Pure accuracy

Have these metrics as well as samples which are different to the ground truth would allow us to further explore where our model needs improvements.


```python
len(dataset["test"]["sequence"])
```


```python
input_test_prompts_formatted = [{"role": "user", "content": item} for item in dataset["test"]["sequence"]]
input_test_prompts_no_format = [item for item in dataset["test"]["sequence"]]
len(input_test_prompts_no_format)
```


```python
# Get preds on the test set
test_outputs = []

for item in tqdm(dataset["test"]):
  prompt = pipe.tokenizer.apply_chat_template(item["messages"][:1], tokenize=False, add_generation_prompt=True)
  outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

  item["test_outputs"] = outputs
  test_outputs.append(item)
```

### Helper functions for condensing and uncondensing items


```python
%%writefile checkpoint_models/helper_functions.py
def condense_output(original_output):
    """Helper function to condense a given FoodExtract string.
    
    Example input: {'is_food_or_drink': True, 'tags': ['fi'], 'food_items': ['cape gooseberries', 'mulberry', 'chilli powder', 'flathead lobster', 'hoisin sauce', 'duck leg', 'chestnuts', 'raw quail', 'duck breast', 'rogan josh curry sauce', 'brown rice', 'dango'], 'drink_items': []}

    Example output: food_or_drink: 1\ntags: fi\nfoods: cape gooseberries, mulberry, chilli powder, flathead lobster, hoisin sauce, duck leg, chestnuts, raw quail, duck breast, rogan josh curry sauce, brown rice, dango\ndrinks:"""

    condensed_output_string_base = '''food_or_drink: <is_food_or_drink>
    tags: <output_tags>
    foods: <food_items>
    drinks: <drink_items>'''

    is_food_or_drink = str(1) if str(original_output["is_food_or_drink"]).lower() == "true" else str(0)
    tags = ", ".join(original_output["tags"]) if len(original_output["tags"]) > 0 else ""
    foods = ", ".join(original_output["food_items"]) if len(original_output["food_items"]) > 0 else ""
    drinks = ", ".join(original_output["drink_items"]) if len(original_output["drink_items"]) > 0 else ""

    condensed_output_string_formatted = condensed_output_string_base.replace("<is_food_or_drink>", is_food_or_drink).replace("<output_tags>", tags).replace("<food_items>", foods).replace("<drink_items>", drinks)

    return condensed_output_string_formatted.strip()

def uncondense_output(condensed_output):
    """Helper to go from condensed output to uncondensed output.

    Example input: food_or_drink: 1\ntags: fi\nfoods: cape gooseberries, mulberry, chilli powder, flathead lobster, hoisin sauce, duck leg, chestnuts, raw quail, duck breast, rogan josh curry sauce, brown rice, dango\ndrinks:

    Example output: {'is_food_or_drink': True, 'tags': ['fi'], 'food_items': ['cape gooseberries', 'mulberry', 'chilli powder', 'flathead lobster', 'hoisin sauce', 'duck leg', 'chestnuts', 'raw quail', 'duck breast', 'rogan josh curry sauce', 'brown rice', 'dango'], 'drink_items': []}
    """

    condensed_list = condensed_output.split("\n")

    condensed_dict_base = {
        "is_food_or_drink": "",
        "tags": [],
        "food_items": [],
        "drink_items": []
    }

    # Set values to defaults
    food_or_drink_item = None
    tags_item = None
    foods_item = None
    drinks_item = None

    # Extract items from condensed_list
    for item in condensed_list:
        if "food_or_drink:" in item.strip():
            food_or_drink_item = item

        if "tags:" in item:
            tags_item = item

        if "foods:" in item:
            foods_item = item

        if "drinks:" in item:
            drinks_item = item

    if food_or_drink_item:
        is_food_or_drink_bool = True if food_or_drink_item.replace("food_or_drink: ", "").strip() == "1" else False
    else:
        is_food_or_drink_bool = None

    if tags_item:
        tags_list = [item.replace("tags: ", "").replace("tags:", "").strip() for item in tags_item.split(", ")]
        tags_list = [item for item in tags_list if item] # Filter for empty items
    else:
        tags_list = []

    if foods_item:
        foods_list = [item.replace("foods:", "").replace("foods: ", "").strip() for item in foods_item.split(", ")]
        foods_list = [item for item in foods_list if item] # Filter for empty items
    else:
        foods_list = []

    if drinks_item:
        drinks_list = [item.replace("drinks:", "").replace("drinks: ", "").strip() for item in drinks_item.split(", ")]
        drinks_list = [item for item in drinks_list if item] # Filter for empty items
    else:
        drinks_list = []

    condensed_dict_base["is_food_or_drink"] = is_food_or_drink_bool
    condensed_dict_base["tags"] = tags_list
    condensed_dict_base["food_items"] = foods_list
    condensed_dict_base["drink_items"] = drinks_list

    return condensed_dict_base
```


```python
# Test them out
import random

random_test_sample = random.choice(test_outputs)
random_test_sequence = random_test_sample["sequence"]
random_test_original_label = random_test_sample["gpt-oss-120b-label"]
random_test_predicted_label_condensed = random_test_sample["test_outputs"][0]["generated_text"].split("<start_of_turn>model")[-1].strip()

print(f"[INFO] Original sequence:")
print(random_test_sequence)
print()
print(f"[INFO] GPT-OSS-120B label:")
print(random_test_original_label)
print()
print(f"[INFO] Uncondensed output:")
print(uncondense_output(random_test_predicted_label_condensed))
print()
print(f"[INFO] Condensed output:")
print(random_test_predicted_label_condensed)
```


```python
for item in test_outputs:
  item["condensed_output"] = uncondense_output(item["test_outputs"][0]["generated_text"].split("<start_of_turn>model")[-1].strip())
```


```python
random_test_sample = random.choice(test_outputs)
random_test_sequence = random_test_sample["sequence"]
random_test_original_label = random_test_sample["gpt-oss-120b-label"]
random_test_predicted_label = random_test_sample["condensed_output"]

print(f"[INFO] Original sequence:")
print(random_test_sequence)
print()
print(f"[INFO] GPT-OSS-120B label:")
print(random_test_original_label)
print()
print(f"[INFO] Uncondensed output:")
print(random_test_predicted_label)
print()
```


```python
# Quick evals

# Get pure equality matching (e.g. strict field to field)
num_bool_matches = 0
num_tags_matches = 0
num_food_items_matches = 0
num_drink_items_matches = 0

for item in test_outputs:
  gpt_labels = eval(item["gpt-oss-120b-label"])
  gemma_labels = item["condensed_output"]

  for key, value in gpt_labels.items():
    is_match = 1 if gemma_labels[key] == value else 0

    if key == "is_food_or_drink":
      num_bool_matches += is_match

    if key == "tags":
      num_tags_matches += is_match

    if key == "food_items":
      num_food_items_matches += is_match

    if key == "drink_items":
      num_drink_items_matches += is_match

print(f"[INFO] Number of direct bool matches: {num_bool_matches}/{len(test_outputs)}")
print(f"[INFO] Number of direct tag matches: {num_tags_matches}/{len(test_outputs)}")
print(f"[INFO] Number of direct food items matches: {num_food_items_matches}/{len(test_outputs)}")
print(f"[INFO] Number of direct drink items matches: {num_drink_items_matches}/{len(test_outputs)}")
```

### More specific evals

Going to expand on the quick evals and see where we get to.

Details:

* For precision and recall we require true positives, false positives and false negatives:
  * **True positives:** Items in both the target set as well as the predicted set.
  * **False positives:** Items in the predicted set but *not* in the target set.
  * **False negatives:** Items in the target set but *not* in the predicted set.


```python
# Get pure equality matching (e.g. strict field to field)
num_bool_matches = 0
num_tags_matches = 0
num_food_items_matches = 0
num_food_items_filter_matches = 0
num_drink_items_matches = 0
num_drink_items_filter_matches = 0

def make_sure_bool_is_str(input):
  if isinstance(input, bool):
    input = str(input).lower()
    input = "true" if "true" in input else "false"
    return input
  else:
    input = str(input).lower()
    input = "true" if "true" in input else "false"
    return input

result_dicts = []
for item in test_outputs:
  result_dict = {}

  gpt_labels = eval(item["gpt-oss-120b-label"])
  gemma_labels = item["condensed_output"]

  result_dict["gpt_labels"] = gpt_labels
  result_dict["gemma_labels"] = gemma_labels
  result_dict["sequence"] = item["sequence"]
  result_dict["uuid"] = item["uuid"]

  # Make sure the types of the bools are the same (we just want all strings)
  gpt_labels["is_food_or_drink"] = make_sure_bool_is_str(input=gpt_labels["is_food_or_drink"])
  gemma_labels["is_food_or_drink"] = make_sure_bool_is_str(input=gemma_labels["is_food_or_drink"])

  for key, value in gpt_labels.items():
    # Get truth labels
    gpt_truth = value
    gemma_truth = gemma_labels[key]

    # Find direct matches
    is_match = 1 if gpt_truth == gemma_truth else 0

    # Go through individual stats and see if there are matches
    if key == "is_food_or_drink":
      result_dict["results_is_food_or_drink_match"] = True if gpt_truth == gemma_truth else False
      num_bool_matches += is_match

    if key == "tags":
      result_dict["results_tags_match"] = True if gpt_truth == gemma_truth else False
      num_tags_matches += is_match

    if key == "food_items":
      result_dict["results_food_items_match"] = True if gpt_truth == gemma_truth else False
      num_food_items_matches += is_match

      # Compare samples with lowering and filtering for uniques
      gpt_food_list_lower = sorted(set([item.lower() for item in gpt_truth]))
      gemma_food_list_lower = sorted(set([item.lower() for item in gemma_truth]))

      # Match filtered and set labels (removes duplicates)
      if gpt_food_list_lower == gemma_food_list_lower:
        num_food_items_filter_matches += 1
        result_dict["result_food_items_filter_match"] = True
      else:
        result_dict["result_food_items_filter_match"] = False

      # Get items that are predicted by Gemma but aren't in the labels (False Positive)
      food_list_false_positive = set(gemma_food_list_lower) - set(gpt_food_list_lower)
      result_dict["result_food_list_false_positive"] = food_list_false_positive

      # Get items that are in GPT labels but aren't predicted by Gemma (False Negative)
      food_list_true_negative = set(gpt_food_list_lower) - set(gemma_food_list_lower)
      result_dict["result_food_list_false_negative"] = food_list_true_negative

    if key == "drink_items":
      result_dict["results_drink_items_match"] = True if gpt_truth == gemma_truth else False
      num_drink_items_matches += is_match

      # Compare samples with lowering and filtering for uniques
      gpt_drink_list_lower = sorted(set([item.lower() for item in gpt_truth]))
      gemma_drink_list_lower = sorted(set([item.lower() for item in gemma_truth]))

      # Match filtered and set labels (removes duplicates)
      if gpt_drink_list_lower == gemma_drink_list_lower:
        num_drink_items_filter_matches += 1
        result_dict["result_drink_items_filter_match"] = True
      else:
        result_dict["result_drink_items_filter_match"] = False

      # Get items that are predicted by Gemma but aren't in the labels (False Positives)
      drink_list_false_positive = set(gemma_drink_list_lower) - set(gpt_drink_list_lower)
      result_dict["result_drink_list_false_positive"] = drink_list_false_positive

      # Get items that are in GPT labels but aren't predicted by Gemma (False Negatives)
      drink_list_true_negative = set(gpt_drink_list_lower) - set(gemma_drink_list_lower)
      result_dict["result_drink_list_false_negative"] = drink_list_true_negative

  result_dicts.append(result_dict)

print(f"[INFO] Number of direct bool matches: {num_bool_matches}/{len(test_outputs)} ({num_bool_matches / len(test_outputs) * 100:.2f}%)")
print(f"[INFO] Number of direct tag matches: {num_tags_matches}/{len(test_outputs)} ({num_tags_matches / len(test_outputs) * 100:.2f}%)")
print(f"[INFO] Number of direct food items matches: {num_food_items_matches}/{len(test_outputs)} ({num_food_items_matches / len(test_outputs) * 100:.2f}%)")
print(f"[INFO] Number of filtered food items matches: {num_food_items_filter_matches}/{len(test_outputs)} ({num_food_items_filter_matches / len(test_outputs) * 100:.2f}%)")
print(f"[INFO] Number of direct drink items matches: {num_drink_items_matches}/{len(test_outputs)} ({num_drink_items_matches / len(test_outputs) * 100:.2f}%)")
```


```python
# Examples
gpt_food = ['wheat flour', 'vegetable oil', 'salt', 'sugar', 'water']
gemma_food = ['wheat flour', 'vegetable oil', 'salt', 'milk', 'vanilla']

def filter_and_lower_list(input_list):
  """Filters a list for uniques and lowers all inputs."""
  if len(input_list) > 0:
    return sorted(set([str(item.lower()) for item in input_list]))
  else:
    return input_list

def calculate_precision_recall_f1(reference, predicted):
  """Calculates precision, recall and F1 scores for two given lists."""

  # Filter items for best comparison
  reference = filter_and_lower_list(input_list=reference)
  predicted = filter_and_lower_list(input_list=predicted)

  # Ensure they're both sets for easy comparison
  ref_set = set(reference)
  pred_set = set(predicted)

  # Handle case where there are empty lists, in this case, the empty predictions are correct
  if (len(ref_set) == 0) and (len(pred_set) == 0):
    precision = 1.0
    recall = 1.0
    f1_score = 1.0

  else:
    # Get TP (True Positives)
    tp = len(ref_set & pred_set) # intersection

    # Get FP (False Positives)
    fp = len(pred_set - ref_set) # items only in predicted but not in reference

    # Get FN (False Negatives)
    fn = len(ref_set - pred_set) # items only in reference but not in predicted

    # Calculate metrics
    # Precision = fp / (tp + fp) - higher precision = less false positives
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall = tp / (tp + fn) - higher recall = less false negatives
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 (combined metric for precision and recall) = 2 * precision * recall / (precision + recall)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

  # Return a dict of metrics
  return {
      "precision": precision,
      "recall": recall,
      "f1_score": f1_score,
      "intersection": list(ref_set.intersection(pred_set)),
      "only_in_reference": list(ref_set - pred_set),
      "only_in_pred": list(pred_set - ref_set)
  }

calculate_precision_recall_f1(reference=gpt_food,
                              predicted=gemma_food)
```

### Calculate precision and recall metrics across all fields


```python
tags_precision = 0
tags_recall = 0
tags_f1 = 0

food_precision = 0
food_recall = 0
food_f1 = 0

drink_recall = 0
drink_precision = 0
drink_f1 = 0

for item in result_dicts:
  tags_metrics = item["tag_metrics"]
  tags_precision += tags_metrics["precision"]
  tags_recall += tags_metrics["recall"]
  tags_f1 += tags_metrics["f1_score"]

  food_metrics = item["food_metrics"]
  food_precision += food_metrics["precision"]
  food_recall += food_metrics["recall"]
  food_f1 += food_metrics["f1_score"]

  drink_metrics = item["drink_metrics"]
  drink_precision += drink_metrics["precision"]
  drink_recall += drink_metrics["recall"]
  drink_f1 += drink_metrics["f1_score"]

print(f"[INFO] Tags metrics:")
print(f"Precision: {tags_precision / len(result_dicts):.3f}")
print(f"Recall: {tags_recall / len(result_dicts):.3f}")
print(f"F1: {tags_f1 / len(result_dicts):.3f}")
print()

print(f"[INFO] Food metrics:")
print(f"Precision: {food_precision / len(result_dicts):.3f}")
print(f"Recall: {food_recall / len(result_dicts):.3f}")
print(f"F1: {food_f1 / len(result_dicts):.3f}")
print()

print(f"[INFO] Drink metrics:")
print(f"Precision: {drink_precision / len(result_dicts):.3f}")
print(f"Recall: {drink_recall / len(result_dicts):.3f}")
print(f"F1: {drink_f1 / len(result_dicts):.3f}")
print()
```
