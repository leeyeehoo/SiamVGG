import argparse
import json

parser = argparse.ArgumentParser(description='label preprocessing')

parser.add_argument('file',metavar='FILE', type=str,
                    help='json file.')

parser.add_argument('output_file',metavar='OUTPUT_FILE', type=str,
                    help='output json file.')

parser.add_argument('path',metavar='PATH', type=str,
                    help='path to change.')

args = parser.parse_args()



if args.file == 'vot2018.txt':
    org_path = '/home/leeyh/Downloads/'#'/home/leeyh/Downloads/vot2017/soldier/00000001.jpg'
elif args.file == 'ilsvrc_vid.txt':
    org_path = '/home/leeyh/Downloads/data/'#'/home/leeyh/Downloads/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00044000/000000.JPEG'
elif args.file == 'youtube_final.txt':
    org_path = '/home/leeyh/youtube-bb/data/youtubebbdevkit2017/'#'/home/leeyh/youtube-bb/data/youtubebbdevkit2017/youtubebb2017/JPEGImages/--0bLFuriZ4+0+0+74000.jpg'

    
    
with open(args.file, 'r') as outfile:
    root = json.load(outfile)
    
for sequence in range(0,len(root)):
    for items in range(0,len(root[sequence])):
        for item in range(0,len(root[sequence][items])):
            root[sequence][items][item][0] = root[sequence][items][item][0].replace(org_path,args.path)

with open(args.output_file, 'w') as outfile:
    json.dump(root, outfile)