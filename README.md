# Object-Centric Explananda via Agent Negotiation

## Installation
```
git clone https://gitlab.doc.ic.ac.uk/bt221/ocean.git
cd ocean
pip install -r requirements.txt
```

## Reproducibility
There are a few example scripts provided. They will work as long as you have install the necessary packages and route the actual datasets in the .env files.
```
./scripts/training/exp_A.sh  # This will run the training for Config A.
```

## Optional Output Folder
```
/out
    /checkpoints
        /sa
            /{epoch}_ckpt.pt
        /cg
            /{epoch}_ckpt.pt
    /.env
    /{metrics}_temp.png
```

## Evaluation
Running the given eval file can be done by providing the path of the out folder.
```
python ./eval.py --latest --reload_path '/path/to/out_folder/' --out_path '/path/to/out_folder/' --top 3
```

## Dialogue Generation
A `dialogue.ipynb` notebook is provided to generate dialogues easily.

## Multi-dSprites
This is optional. We generated an instance of the Multi-dSprites dataset and augmented with classification labels. To generate your own instance, you need to install the record file from <https://github.com/google-deepmind/multi_object_datasets>.
```
wget https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords
wget -P ./scripts/datasets https://raw.githubusercontent.com/google-deepmind/multi_object_datasets/refs/heads/master/multi_dsprites.py
pip install pandas tensorflow==2.16.1
python ./scripts/datasets/generate_mds.py --record_path /path/to/record --save_path /path/to/save/dataset
```