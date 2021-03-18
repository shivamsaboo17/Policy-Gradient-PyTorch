# Policy-Gradient-PyTorch
Implementation of vanilla stochaistic (categorical) policy gradient algorithm to play cartpole. </br>
Vanilla policy gradient takes longer but convergence is smoother than DQN for the cartpole, both of these properties as expected.</br></br>

## Training
```bash
python ./vpg_pytorch.py 
```

## Testing
```bash
python test_vpg_pytorch.py <directory>
python test_vpg_pytorch.py "Save/2021-03-18-15:42:31"
```
This test every saved weight file (vpg_weight_ep*.pth) under the directory  for 100 episodes.


## Running with rendering
```bash
./run_learnt_model.py <file>
./run_learnt_model.py "Save/2021-03-18-15:42:31/vpg_weight_ep9970.pth"
```
This executes the cartpole env with rendering and show you how the learnt model actually works.
