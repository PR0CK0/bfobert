import pandas as pd, torch, warnings, psutil
from datetime import datetime
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
  DistilBertTokenizerFast,
  DistilBertForSequenceClassification,
  Trainer,
  TrainingArguments,
  TrainerCallback,
  EarlyStoppingCallback
)

############################################################

class ResourceMonitorCallback(TrainerCallback):
  def __init__(self):
    self.stats = []

  def on_epoch_end(self, args, state, control, **kwargs):
    memory = psutil.virtual_memory().used / (1024 ** 3)
    cpu = psutil.cpu_percent(interval = 0.5)
    entry = {
      'epoch': state.epoch,
      'cpu_percent': cpu,
      'mem_gb': memory,
      'timestamp': datetime.now().strftime('%H:%M:%S')
    }
    self.stats.append(entry)
    print(f"[Resource] Epoch {state.epoch:.1f} | CPU {cpu:.1f}% | RAM {memory:.2f} GB")

  def on_train_end(self, args, state, control, **kwargs):
    print("\n=== Resource Summary ===")
    for e in self.stats:
      print(f"Epoch {e['epoch']:.1f}: CPU {e['cpu_percent']:.1f}% | RAM {e['mem_gb']:.2f} GB at {e['timestamp']}")

############################################################

class BFOBERTTrainer:
  def __init__(self, dataPath = 'terms.csv'):
    warnings.filterwarnings('ignore', message = '.*pin_memory.*')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', self.device)

    self.dataPath = dataPath
    self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    self.dataset = None
    self.model = None
    self.trainer = None

  def loadDataset(self):
    df = pd.read_csv(self.dataPath).sample(frac = 1).reset_index(drop = True)
    ds = Dataset.from_pandas(df.rename(columns = {'term': 'text', 'label': 'labels'}))
    ds = ds.train_test_split(test_size = 0.2)

    def tokenizeBatch(batch):
      return self.tokenizer(
        batch['text'],
        truncation = True,
        padding = 'max_length',
        max_length = 32
      )

    ds = ds.map(tokenizeBatch, batched = True)
    ds.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
    self.dataset = ds
    print(ds)

  def initializeTwoHeadModel(self):
    model = DistilBertForSequenceClassification.from_pretrained(
      'distilbert-base-uncased',
      num_labels = 2
    ).to(self.device)

    self.freezeAllLayers(model)
    self.unfreezeLastTwoLayers(model)

    print('Trainable parameters:')
    for n, p in model.named_parameters():
      if p.requires_grad:
        print(' ', n)

    self.model = model
    
  def freezeAllLayers(self, model):
    for param in model.distilbert.parameters():
      param.requires_grad = False

  def unfreezeLastTwoLayers(self, model):
    for name, param in list(model.distilbert.transformer.layer.named_parameters())[-2:]:
      param.requires_grad = True

  @staticmethod
  def computeMetrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
      labels, preds, average = 'binary'
    )
    acc = accuracy_score(labels, preds)

    try:
      auc = roc_auc_score(labels, pred.predictions[:, 1])
    except ValueError:
      auc = float('nan')

    return {
      'accuracy': acc,
      'precision': precision,
      'recall': recall,
      'f1': f1,
      'roc_auc': auc
    }

  def configureTrainer(self):
    args = TrainingArguments(
      output_dir = f'./results/{self.timestamp}',
      eval_strategy = 'epoch',
      save_strategy = 'epoch',
      load_best_model_at_end = True,
      num_train_epochs = 6,
      per_device_train_batch_size = 16,
      per_device_eval_batch_size = 16,
      learning_rate = 4e-5,
      weight_decay = 0.01,
      logging_dir = './logs',
      logging_strategy = 'steps',
      logging_steps = 50,
      metric_for_best_model = 'f1',
      greater_is_better = True,
      report_to = ['tensorboard']
    )

    self.trainer = Trainer(
      model = self.model,
      args = args,
      train_dataset = self.dataset['train'],
      eval_dataset = self.dataset['test'],
      tokenizer = self.tokenizer,
      compute_metrics = BFOBERTTrainer.computeMetrics,
      callbacks = [
        ResourceMonitorCallback(),
        EarlyStoppingCallback(
          early_stopping_patience = 2,
          early_stopping_threshold = 0.01
        )
      ]
    )

  def train(self):
    self.trainer.train()
    print('\nFinal evaluation:')
    self.trainer.evaluate()

  def classifySamples(self, samples):
    labels = ['continuant', 'occurrent']
    tokenized = self.tokenizer(samples, truncation = True, padding = True, max_length = 32, return_tensors = 'pt').to(self.device)

    with torch.no_grad():
      outputs = self.model(**tokenized)
      probs = torch.nn.functional.softmax(outputs.logits, dim = 1)
      classes = probs.argmax(dim = 1).cpu().numpy()

    for term, cls, prob in zip(samples, classes, probs):
      print(f'{term:<25} -> {labels[cls]}  ({prob[cls]:.3f} confidence)')

############################################################

if __name__ == '__main__':
  clf = BFOBERTTrainer('terms.csv')
  clf.loadDataset()
  clf.initializeTwoHeadModel()
  clf.configureTrainer()
  clf.train()
  clf.classifySamples([
    'ancient bridge',
    'rapid growth',
    'blue cell',
    'chemical reaction',
    'mechanical engine',
    'dance'
  ])