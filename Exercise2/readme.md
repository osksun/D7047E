# Answers to practical part
2.1 Transfer Learning from ImageNet

AlexNet accuracy on CIFAR10: 13.890%

AlexNet accuracy on CIFAR10 using pretrained model: 86.750% 

The difference is that in the first run we trained the model ourselves on the CIFAR10 dataset with untrained neurons
while the second run downloads a pretrained model and trains on neurons that has already been trained on before.

2.2 Transfer Learning from MNIST

CNN_basic accuracy on SVHN: 14,916%

CNN_basic accuracy on MNIST: 88,39%

CNN_basic trained on MNIST loaded and trained on SVHN: 30,032%


# Commands and parts of logs used for this exercise
```sh
# Train AlexNet on CIFAR10 with seed=42.
python template/RunMe.py --model-name alexnet --output-folder exercise_2_log --dataset-folder toy_datasets/CIFAR10 --seed 42 --ignoregit --no-cuda
# Result:
#[    INFO] --- test  epoch[4]: Acc@1=14.080	Loss=2.3021	Batch time=0.343 (0.005 to load data) (evaluate.py:141)
# ~14.08% accuracy on test set.

# Now train again, with the same setting but this time you will add the flag -pretrained.
python template/RunMe.py --model-name alexnet --pretrained --output-folder exercise_2_log --dataset-folder toy_datasets/CIFAR10 --seed 42 --ignoregit --no-cuda
# Result:
#[    INFO] --- test  epoch[4]: Acc@1=86.250    Loss=0.3818 Batch time=0.342 (0.004 to load data) (evaluate.py:141)
# ~86.25% accuracy on test set.

# (1) Train CNN_basic on SVHN.
python template/RunMe.py --model-name CNN_basic --output-folder exercise_2_log --dataset-folder toy_datasets/SVHN --seed 42 --ignoregit
#[    INFO] --- test  epoch[4]: Acc@1=14.909    Loss=2.2945 Batch time=0.011 (0.001 to load data) (evaluate.py:141)
# Result: ~14.91% accuracy on test set.

# (2) Train CNN_basic on MNIST.
python template/RunMe.py --model-name CNN_basic --output-folder exercise_2_log --dataset-folder toy_datasets/MNIST --seed 42 --ignoregit
# Result:
#[    INFO] --- test  epoch[4]: Acc@1=88.390    Loss=0.4146 Batch time=0.094 (0.082 to load data) (evaluate.py:141)
# 88.39% accuracy on test set.

# Load the model from (2) and then train it (fine tune) on SVHN.
python template/RunMe.py --model-name CNN_basic --output-folder exercise_2_log --dataset-folder toy_datasets/SVHN --seed 42 --ignoregit --load-model exercise_2_log/mnist_CNN_basic/MNIST/seed=42/07-04-20-18h-14m-21s/checkpoint.pth.tar
# Result:
#[    INFO] --- test  epoch[4]: Acc@1=28.857    Loss=2.1238 Batch time=0.011 (0.001 to load data) (evaluate.py:141)
# ~28.56% accuracy on test.
```
