"""Writes overfit_check_colab.ipynb — can PhonoGCN memorise 32 training clips?

If WER on those same 32 clips drops well below 1.0, the model and training
code can learn and the full run needs more steps / a higher LR. If it stays
at 1.0, something in the model or loss is broken and needs debugging first.
Uses the phonogcn_colab.zip already in My Drive.
"""
import json, sys

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
    md("# PhonoGCN — can the model learn at all? (about 30 minutes)\n\n"
       "Trains on just **32 clips** and then tests on **the same 32 clips**. A model that works should "
       "memorise them, so WER should drop well below 1.0.\n\n"
       "Run the cells **one at a time, top to bottom** (Shift + Enter). "
       "First: *Runtime → Change runtime type → T4 GPU → Save*."),
    md("## 1. Check the GPU"),
    code("!nvidia-smi"),
    md("## 2. Connect Google Drive (click **Allow** in the pop-up)"),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    md("## 3. Unpack the bundle (skip if you already did it in this session)"),
    code("import os\n"
         "ZIP = '/content/drive/MyDrive/phonogcn_colab.zip'\n"
         "assert os.path.exists(ZIP), 'phonogcn_colab.zip not found in the top level of My Drive'\n"
         "if not os.path.exists('/content/phonogcn_colab'):\n"
         "    !cp \"$ZIP\" /content/\n"
         "    !unzip -q -o /content/phonogcn_colab.zip -d /content/\n"
         "print('ready')"),
    md("## 4. Make the 32-clip set\nTrain, validation and test all use the same 32 training clips."),
    code("import json\n"
         "D = '/content/phonogcn_colab/data/KSL_Daily_500_real_10fps'\n"
         "O = '/content/overfit32'\n"
         "os.makedirs(O, exist_ok=True)\n"
         "clips = json.load(open(D + '/train_manifest.json'))[:32]\n"
         "for s in ('train', 'val', 'test'):\n"
         "    json.dump(clips, open(f'{O}/{s}_manifest.json', 'w'))\n"
         "for x in ('skeletons', 'crops', 'gloss_vocabulary.json'):\n"
         "    if not os.path.exists(f'{O}/{x}'):\n"
         "        os.symlink(f'{D}/{x}', f'{O}/{x}')\n"
         "print(len(clips), 'clips,', sum(len(c['gloss_ids']) for c in clips), 'gloss labels')"),
    md("## 5. Train on the 32 clips (150 epochs, learning rate 1e-3)\n"
       "Watch `val_wer` in the `Epoch …` lines. **Good sign:** it falls below 1.0 and keeps going down. "
       "**Bad sign:** it stays at exactly 1.0000 to the end."),
    code("os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
         "OUT = '/content/drive/MyDrive/phonogcn_results/overfit32'\n"
         "os.makedirs(OUT, exist_ok=True)\n"
         "%cd /content/phonogcn_colab/code\n"
         "!python train.py --data_root /content/overfit32 --output_dir \"$OUT\" --device cuda --num_workers 2 \\\n"
         "    --config_override train.epochs=150 train.n_seeds=1 train.batch_size=8 train.lr=0.001 \\\n"
         "    train.save_every_n_epochs=1000 data.n_frames_max=160 2>&1 | tee \"$OUT/train_log.txt\""),
    md("## 6. Result"),
    code("print(open(OUT + '/results.json').read())"),
    md("Done. Download **My Drive → phonogcn_results → overfit32** (only `train_log.txt` and `results.json` are needed; "
       "you can skip the `.pt` file) and share it."),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("wrote", sys.argv[1])
