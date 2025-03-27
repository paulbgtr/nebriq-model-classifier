# nebriq-model-classifier 🧠

A custom transformer-based classifier that predicts the **semantic complexity** of user prompts — trained to help intelligently route prompts to the most suitable LLM (Large Language Model).

## ✨ What is this?

This is the model selection brain behind **[Nebriq](https://nebriq.com)** — a minimalist, AI-powered note-taking app with a strong focus on AI-assisted workflows.

Instead of hardcoding logic like:

```python
if "analyze" in prompt:
    use("gpt-4")
```

…this classifier understands the prompt and determines whether it’s:

- simple: small talk, casual, or factual
- medium: instructional, contextual, or practical
- advanced: abstract, creative, analytical, or deeply technical

## 🧩 Why does it exist?

Users shouldn’t need to know which model is “best” for a task.

This classifier allows Nebriq to automatically route user prompts to the right LLM backend (e.g., GPT-4o-mini, Claude 3, DeepSeek, etc.) depending on semantic intent and complexity, making the UX faster, cheaper, and smarter.

## 🔧 How was it built?

- Trained on a small curated dataset of user-like prompts
- Built using 🤗 Transformers (distilbert-base-uncased)
- Uses the Hugging Face Trainer API
- Includes support for inference, training, and evaluation

## 💻 Example Inference

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="paulbg/nebriq-model-classifier")

prompt = "Model the economic effects of inflation on developing countries"
res = classifier(prompt)
print(res)
# → [{'label': 'advanced', 'score': 0.94}]
```

## 🛠️ Local Setup

```bash
poetry install
poetry run python scripts/train.py # for training
poetry run python scripts/inference.py # for inference
```

## 🤝 Use cases

- Prompt routing inside AI-native tools (like Nebriq)
- Custom moderation / filter / rerouting layers
- “AI load balancer” for hybrid LLM backends

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

🤗 [View on Hugging Face](https://huggingface.co/paulbg/nebriq-model-classifier)
