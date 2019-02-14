# SiamVGG

SiamVGG adopts SiamFC as the baseline approach. It applies a fully-convolutional Siamese network to allocate the target in the search region. SiamVGG uses modified VGG-16 network as the backbone. The network is trained offline on both ILSVRC VID dataset and Youtube-BB dataset end-to-end. The model SiamVGG gets significant improvements when compare to the original SiamFC.

## Getting Started

### Hardware Environment

* System: ubuntu 14.04 LTS

* Memory: 15.6 GiB

* Processor: Intel Core i7-7700K CPU @4.20GHz * 8

* Graphics: GeForce GTX 1080 Ti/PCle/SSE2

* OS type: 64-bit

### Prerequisites

* CUDA compilation tools version: release 8.0, V8.0.61

* NVIDIA driver version: 390.59

* MATLAB version: 9.3.0.713579 (R2017b)

* Please use Anaconda environment for the test
  
  * python                    2.7.13
  
  * pytorch                   0.2.0
  
  * torchvision               0.1.8
  
  * pillow                    4.1.1
  
  * numpy                     1.12.1
  
  * hdf5                      1.8.17
  
### Preparation for the tests

1. We assume the vot-toolkit root is VOT-TOOLKIT, and `$VOT-TOOLKIT = /path/to/vot-toolkit`

```

git clone https://github.com/votchallenge/vot-toolkit.git

```

2. add `$VOT-TOOLKIT` and its subfolder to MATLAB path

3. create `$VOT-TOOLKIT/vot-workspace`

4. run `workspace_create.m` in `$VOT-TOOLKIT/vot-workspace`, assume your tracker as 'SiamVGG' with python interpreter.

5. replace `$VOT-TOOLKIT/tracker/examples/python` with `tracker/examples/python`

6. replace `$VOT-TOOLKIT/vot-workspace／tracker_SiamVGG.m` with `vot-workspace/tracker_SiamVGG.m`

7. modify `line 17` in `$VOT-TOOLKIT/vot-workspace／tracker_SiamVGG.m`

```

tracker_command = generate_python_command('python_siamvgg', {'$VOT-TOOLKIT/tracker/examples/python'});

```

8. modify `line 8`, `line 10`, `line 12`, `line 14`, `line 16` in `$VOT-TOOLKIT/tracker/examples/python/src/parse_arguments.py`, replace `/home/lee/tracking/challenge/vot-toolkit/` with `$VOT-TOOLKIT/`


9. modify `line 45` in `$VOT-TOOLKIT/tracker/examples/python/src/siamvggtracker.py`

```

NET_PATH = '$VOT-TOOLKIT/tracker/examples/python/pretrained/000100vggv1net1-5.weights'

```
### Results

The results should be the same on VOT2018 dataset.

|               | Baseline           | Long-term  |
| ------------- |:-------------:| -----:|
| EAO      | 0.285 | - |
| TPR      | - | 0.459 |
