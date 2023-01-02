import Dataset
from Train import fit , evaluate

print("*****Training*****")
fit(Dataset.train_dataloader)

evaluate()