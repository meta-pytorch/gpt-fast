export MODEL_REPO=mistralai/Mixtral-8x7B-Instruct-v0.1

# python generate.py --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth
python generate.py --compile --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth
