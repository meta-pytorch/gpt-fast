export MODEL_REPO=mistralai/Mixtral-8x7B-Instruct-v0.1

# python generate.py --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth
# echo "1"
# echo "python generate.py --compile --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --batch_size 4
# "
# python generate.py --compile --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --batch_size 4
# echo "2"
# echo "python generate.py --compile --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --batch_size 4
# "
# python generate.py --compile --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --batch_size 4
# echo "3"
# echo "python generate.py --compile --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --batch_size 4
# "
# python generate.py --compile --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --batch_size 4

python generate.py --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --compile --profile no_q_new_model


# python generate.py --checkpoint_path ../checkpoints/$MODEL_REPO/model.pth --batch_size 4 --compile --profile "no_q_profile"

# quant reduced layers
# Time for inference 2: 2.25 sec total, 88.82 tokens/sec
# Bandwidth achieved: 8296.64 GB/s
# Average tokens/sec: 163.61
# Memory used: 94.12 GB
