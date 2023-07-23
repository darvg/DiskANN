# Random Label Generation

These scripts are designed for generating random labels for use with DiskANN across a variety of distributions.

## Installation
```
cd label_gen
pip install -r requirements.txt
pip install -e base
```

## Usage
```
cd label_gen/scripts
python3 label_gen_classes.py ...
python3 label_stats.py ...
```
### label_gen_classes.py
Before generating labels with `label_gen_classes.py`, we need to add classes to the file to pick from.
