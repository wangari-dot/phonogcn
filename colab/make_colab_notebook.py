"""Writes run_phonogcn_colab.ipynb — the notebook that trains PhonoGCN on a Colab GPU."""
import json, sys

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
    md("# PhonoGCN — training on real KSL clips (Colab GPU)\n\n"
       "Run the cells **one at a time, top to bottom** (click a cell, then press **Shift + Enter**).\n\n"
       "Before you start: *Runtime → Change runtime type → T4 GPU → Save*."),
    md("## 1. Check the GPU\nYou should see a table that mentions **Tesla T4** (or another NVIDIA GPU). "
       "If you see an error instead, the GPU is not switched on — go back to *Runtime → Change runtime type*."),
    code("!nvidia-smi"),
    md("## 2. Connect Google Drive\nA pop-up will ask for permission — choose your Google account and click **Allow**."),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    md("## 3. Unpack the bundle\nCopies `phonogcn_colab.zip` from the top level of *My Drive* to this machine and unzips it (a few minutes)."),
    code("import os\n"
         "ZIP = '/content/drive/MyDrive/phonogcn_colab.zip'\n"
         "assert os.path.exists(ZIP), 'phonogcn_colab.zip not found in the top level of My Drive'\n"
         "!cp \"$ZIP\" /content/\n"
         "!unzip -q -o /content/phonogcn_colab.zip -d /content/\n"
         "!ls /content/phonogcn_colab /content/phonogcn_colab/data/KSL_Daily_500_real_10fps"),
    md("## 4. Train (20 epochs, 1 seed)\n"
       "Results are written straight to **My Drive → phonogcn_results**, so nothing is lost if Colab disconnects.\n\n"
       "A line like `Epoch 3/20 loss=... val_wer=...` appears after each epoch. Leave this tab open while it runs — if Colab disconnects, run cells 2–4 again (training restarts from epoch 1). If you get **`CUDA out of memory`**, change `BATCH = 8` to `BATCH = 4` and run the cell again."),
    code("BATCH = 8\n"
         "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
         "OUT = '/content/drive/MyDrive/phonogcn_results/real_10fps_20ep'\n"
         "os.makedirs(OUT, exist_ok=True)\n"
         "%cd /content/phonogcn_colab/code\n"
         "!python train.py --data_root ../data/KSL_Daily_500_real_10fps --output_dir \"$OUT\" --device cuda --num_workers 2 \\\n"
         "    --config_override train.epochs=20 train.n_seeds=1 train.batch_size=$BATCH data.n_frames_max=160 2>&1 | tee \"$OUT/train_log.txt\""),
    md("## 5. See the result\nShows the final numbers. WER = word error rate (lower is better; 1.0 means nothing was recognised)."),
    code("print(open(OUT + '/results.json').read())"),
    md("Done. Everything (log, `results.json`, checkpoints) is in **My Drive → phonogcn_results → real_10fps_20ep**. "
       "Download that folder and put it in `PhonoGCN model/` on your computer."),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("wrote", sys.argv[1])
