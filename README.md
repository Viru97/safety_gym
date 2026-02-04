## Prepare env
```bash
conda create -n safety python=3.8
conda activate safety
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

```

```bash
# test cuda
python -c "import torch; print(torch.cuda.is_available())"
```

## Before start, test with pygame to better understand the task

```bash
cd safety_gym
python envs/safe_stop_v1.py
# a window will pop up, use arrow key to control the ego robot to avoid collision.
# you can use default objects spawning logic as problem difficulty. or design by yourown. 

```

## Start training and evaluation
```bash
python safe_stop_train.py --mode train --n_envs 256
python safe_stop_train.py --mode eval --model_path ./logs/*/model_*.pt --n_envs 1 --render
```

## Demo requirement
- run online redering with your trained model. 
- explain training methods and analysis of key findings during this process.
- any interesting observation and idea will be encouraged. 
- training results is not only metric to evaluate the work, but understanding of the problem, and analysis which can help you dive into probelm is highly expected. 