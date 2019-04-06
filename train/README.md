**Dataset**
[ILSVRC](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php)
[YOUTUBE-BB](https://github.com/mbuckler/youtube-bb)

**Preprocessing**

Please Download the [zip file](https://drive.google.com/open?id=13aC_2stCEU0VoiIpp6wgudenUSVye74b) if you do not want to generate json files yourself. It includes three json files: `ilsvrc_vid.txt`, `youtube_final.txt`, `vot2018.txt`. In our training process, we use `ilsvrc_vid.txt` and `youtube_final.txt` to train and `vot2018.txt` to verify the model.

Try `python label_preprocess.py FILE OUTPUTFILE PATH` to do the preprocess for your `*.txt`. For example, if you download the VOT2018 dataset in the folder `/root/myfolder/` and the original file is `vot2018.txt` and the output file is `vot2018_new.txt`, then your command should be `python label_preprocess.py vot2018.txt vot2018_new.txt /root/myfolder/`

**Training**

Firstly, we suppose you download the `ILSVRC` and `YOUTUBE` dataset both. If you only have ILSVRC dataset, you should modify the `dataset.py` to remove the `YOUTUBE` part.

Try `python train.py GPU TASK --pre='PATH_TO_SAVED_MODEL'`. Choose `GPU` as `0` if you only have one GPU (currently we don't suppose multi GPUs). `TASK` is the name of your saved file. For example, if your save a model in `backup/bestSIAMVGG.pth.tar` and you want to train on your single GPU and the task name is `SIAMVGG`, your command should be `python train.py 0 SIAMVGG --pre='backup/bestSIAMVGG.pth.tar'`.
