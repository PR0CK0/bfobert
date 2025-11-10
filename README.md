# BFO-BERT Classifier

A lightweight fine-tuning script for **DistilBERT** that classifies ontology terms as either **Continuant (0)** or **Occurrent (1)** in the spirit of the **Basic Formal Ontology (BFO)**. Designed as a simple, reproducible proof of concept for text-based ontological classification.

---

## 🧩 Features

- Fine-tunes a small BERT model (`distilbert-base-uncased`) for binary classification
- Automatically logs CPU and RAM usage each epoch
- Early stopping when F1 score stops improving
- TensorBoard-compatible logging
- Timestamped output folders under `/results` and `/logs`
- Quick inference on arbitrary term samples

## ⚙️ Dependencies

Install these Python packages (Python 3.9+ recommended):

```bash
pip install torch
pip install transformers==4.57.1
pip install datasets
pip install scikit-learn
pip install pandas
pip install psutil
pip install tensorboard
pip install accelerate
```

## 📁 Project Structure

```bash
bfobert/
├── bfobert.py          # main training script
├── randomTerms.py      # synthetic fine-tune dataset generator
├── terms.csv           # dataset (columns: term, label)
├── logs/               # tensorboard logs (auto-created)
├── results/            # model outputs (auto-created)
├── venv/               # virtual environment (ignored)
├── .gitignore
└── README.md
```

## ▶️ How to Run

From the console:

```python bfobert.py```


or, if using a virtual environment:

```bash
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
python bfobert.py
```

## 📊 After Running

* Results and checkpoints appear in ./results/_timestamp_/
* Logs for TensorBoard in ./logs/
* Launch TensorBoard to visualize training metrics:
  * ```tensorboard --logdir ./logs```
* Final CPU/RAM summary is printed to console

## 🧠 Example Output

```bash
Using device: cuda
Trainable parameters:
  transformer.layer.5.attention.q_lin.weight
  transformer.layer.5.output_layer_norm.bias
[Resource] Epoch 1.0 | CPU 82.4% | RAM 7.62 GB
...
=== Resource Summary ===
Epoch 1.0: CPU 82.4% | RAM 7.62 GB
Epoch 2.0: CPU 84.0% | RAM 7.89 GB

ancient bridge          -> continuant  (0.784 confidence)
chemical reaction       -> occurrent   (0.698 confidence)
```

🧩 Dataset Format

Example terms.csv:

| term	| label |
| - | - |
| river stone	| 0 |
| chemical reaction | 1 |
|wooden chair |	0 |
| combustion | 1 |

* 0 → Continuant
* 1 → Occurrent

## 💡 Notes

* Not modular, not written for easy change of values... just a nice little test of ontology classification
* Adjust model freezing logic in initializeModel() if you want to fine-tune more or fewer layers
* You can modify dataset size or experiment with other small Transformer backbones, e.g. MiniLM, BERT-tiny
* Each run produces its own timestamped directory under /results and /logs, making experiment tracking easy
* You can view resource usage summaries directly in the console at the end of training