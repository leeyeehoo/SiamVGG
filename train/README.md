**preprocessing**


Please Download the [zip file](https://drive.google.com/open?id=13aC_2stCEU0VoiIpp6wgudenUSVye74b) if you do not want to generate json files yourself. It includes three json files: `ilsvrc_vid.txt`, `youtube_final.txt`, `vot2018.txt`. In our training process, we use `ilsvrc_vid.txt` and `youtube_final.txt` to train and `vot2018.txt` to verify the model.

try `python label_preprocess.py FILE OUTPUTFILE PATH` to do the preprocess for your `*.txt`. For example, if you download the VOT2018 dataset in the folder `root/myfolder/` and the original file is `vot2018.txt` and the output file is `vot2018_new.txt`, then your command should be `python label_preprocess.py vot2018.txt vot2018_new.txt root/myfolder/`
