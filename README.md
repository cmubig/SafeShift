# SafeShift: Safety-Informed Distribution Shifts for Robust Trajectory Prediction in Autonomous Driving 

As autonomous driving technology matures, the safety and robustness of its key components, including trajectory prediction is vital. Although real-world datasets such as Waymo Open Motion provide recorded real scenarios, the majority of the scenes appear benign, often lacking diverse safety-critical situations that are essential for developing robust models against nuanced risks. However, generating safety-critical data using simulation faces severe simulation to real gap. Using real-world environments is even less desirable due to safety risks. In this context, we propose an approach to utilize existing real-world datasets by identifying safety-relevant scenarios naively overlooked, e.g., near misses and proactive maneuvers. Our approach expands the spectrum of safety-relevance, allowing us to study trajectory prediction models under a safety-informed, distribution shift setting. We contribute a versatile scenario characterization method, a novel scoring scheme for reevaluating a scene using counterfactual scenarios to find hidden risky scenarios, and an evaluation of trajectory prediction models in this setting. We further contribute a remediation strategy, achieving a 10% average reduction in predicted trajectories' collision rates. To facilitate future research, we release our code for this overall SafeShift framework to the public: github.com/cmubig/SafeShift

Repository based off of MTR: https://github.com/sshaoshuai/MTR  

## Installation 

- Create and activate a virtual environment on Python 3.8: `conda create -n safeshift python=3.8; conda activate safeshift`
- Clone and enter this repository locally: `git clone https://github.com/cmubig/SafeShift.git ; cd SafeShift`
- Install initial requirements:
    - `python -m pip install -r requirements.txt`
- Refer to the configuration specified at [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/) for torch 1.13, e.g.:
    - `python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`
- Final package installations:
    - `python -m pip install pandas numpy waymo-open-dataset-tf-2-6-0`
    - `python -m pip install -e .`

## Waymo Dataset Preparation

- Download the `scenario protocol` files from the Waymo Open Motion Dataset, v1.2.0, from [https://waymo.com/open/download/](https://waymo.com/open/download/) .
    - e.g., store the raw `scenario` files as `~/waymo/scenario/training`, `~/waymo/scenario/validation`, etc.
- Run preprocessing on the raw files, where the first argument is the raw input and second is the output directory:
    - `cd mtr/datasets/waymo`
    - `python data_preprocess.py ~/waymo/scenario ~/waymo/mtr_process`
    - `cd ../../..`

### Creating dataset splits

- Now, run `resplit` to create a new **uniform** train/val/test split based on the original train/val
    - `cd tools; python resplit.py`
- For convenience, we now `mv` all of the processed files into a single, `joint_original` folder
    - ```
        mkdir ~/waymo/mtr_process/joint_original
        for f in training validation testing; do cp ~/waymo/mtr_process/new_processed_scenarios_${f}/*.pkl ~/waymo/mtr_process/joint_original/. ; done
        for f in training validation testing; do rm -rf ~/waymo/mtr_process/new_processed_scenarios_${f} ~/waymo/mtr_process/processed_scenarios_${f} ; done
      ```
- Download and place all `pkl` files from [https://cmu.box.com/s/ptl5vlsi5uwt6drejnrpcp8a9utfwuzo](https://cmu.box.com/s/ptl5vlsi5uwt6drejnrpcp8a9utfwuzo) into the same `~/waymo/mtr_process` directory

### Scenario Characterization

- Instructions to be provided later; for now, utilize the above Box link to download split meta files

## Training and Evaluation

- All training and evaluation in the `tools` directory
- The main pre-built config files are `tools/cfgs/mini*`
    - Uniform split: `tools/cfgs/mini/mtr+20p_64_a.yaml`
    - Clusters split: `tools/cfgs/mini_frenet_013/mtr+20p_64_a.yaml`
    - Scoring split (no remediation): `tools/cfgs/mini_score_asym_combined/mtr+20p_64_a.yaml`
    - Scoring split (remediation): `tools/cfgs/mini_weighted_score_asym_combined/mtr+20p_64_a.yaml`

### Training 
For example, train with 4 GPUs: 
```
bash scripts/dist_train.sh 4 --cfg_file cfgs/mini/mtr+20p_64_a.yaml --batch_size 32 --extra_tag default
```
- During the training process, the evaluation results will be logged to the log file under `output/mini/mtr_20p_64_a_default/log_train_xxxx.txt`
- Feel free to add the `--no_ckpt` flag to restart the training from scratch for the same tag

### Using pre-trained weights

- If you are given pretrained weights, in the form of `*.pth` files, perform a dry-run of the above training script by letting it start to train for a minute or two, then terminate the process
- Next, place the files in the appropriate folder, e.g. `mini/mtr+20p_64_a/default/ckpt`, and proceed to Testing section of this document

### Testing
For example, test with 4 GPUs: 
```
bash scripts/dist_test.sh 4 --cfg_file cfgs/mini/mtr+20p_64_a_test.yaml --ckpt ../output/mini/mtr+20p_64_a/default/ckpt/best_model.pth --save_train --save_val --batch_size 32
```
- This will create output in the corresponding `output/mini/mtr+20p_64_a_test` directory
    - `log_eval_*.txt` will have output in train, val, test order
- You can then run an additional processing script, `metric_res.py` for detailed analysis:
    - `python metric_res.py --cfg_file cfgs/mini/mtr+20p_64_a_test.yaml --parallel --nproc 20`
    - `python metric_res.py --cfg_file cfgs/mini/mtr+20p_64_a_test.yaml --parallel --nproc 20 --gt`
- This populates the `log_metrics_*.txt` and `log_gt_metrics_*.txt` files respectively,
  where the latter corresponds to replacing the predicted future trajectories with ground truth futures.
