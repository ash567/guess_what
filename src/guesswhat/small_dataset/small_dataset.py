"""Create a smaller dataset (images + games) from specified GuessWhat dataset

example
-------
python src/guesswhat/preprocess_data/small_dataset.py -data_dir=/path/to/guesswhat -no_examples=number of examples

eg:

 python /home/ishu/guesswhat/src/guesswhat/small_dataset/small_dataset.py -data_dir /home/ishu/guesswhat/data -no_examples 300

"""

import argparse
import collections
import io
import json
import os

from guesswhat.data_provider.guesswhat_dataset import OracleDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating Small Dataset')
    parser.add_argument("-data_dir", type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-no_examples", type=int, default=333, help="Number of examples")

    args = parser.parse_args()
    options = [(2, "train"), (1, "test"), (1, "valid")]
    path = args.data_dir + "/guesswhat."
    cwd = os.getcwd()

    for (n, s) in options:
        print
        print s
        print "****"
        s_big = s + "_big"
        cmd = []

        if os.path.isfile(path  + s_big + ".jsonl.gz") and os.path.isfile(path  + s + ".jsonl.gz"):
            print "Small dataset already present"
            print
            print  "----------------"
            continue

        # Unzipping the dataset
        cmd.append("gunzip " + path  + s + ".jsonl.gz")
        # renaming the true large dataset
        cmd.append("mv " + path  + s + ".jsonl " + path  + s_big + ".jsonl")
        # Making a new small dataset (Taking only some of the games)
        cmd.append("head -%d " % (n * args.no_examples) + path  + s_big + ".jsonl > " + path + s + ".jsonl")

        # Zipping back the files
        cmd.append("gzip " + path  + s_big + ".jsonl")
        cmd.append("gzip " + path  + s + ".jsonl")

        for c in cmd:
            print c
            os.system(c)
        print
        print  "----------------"


    print "Printing the image files"
    img_dict = {}
    for (n, s) in options:
        game_set = OracleDataset.load(args.data_dir, s)
        for game in game_set.games:
            img_dict[str(game.image.id) + ".jpg"] = 1

    # print img_dict

    if not os.path.exists(args.data_dir + "/img/raw_big"):
        os.makedirs(args.data_dir + "/img/raw_big")

    cmd_string = []
    i = 0
    for file in os.listdir(args.data_dir + "/img/raw"):
        if file.endswith(".jpg"):
            if file in img_dict:
                i = i + 1
                continue
            # Moving files that are not in the game to other folder
            cmd_string.append("mv " + args.data_dir + "/img/raw/" + file + " " + args.data_dir + "/img/raw_big/" + file)
            # print cmd_string
    print len(cmd_string) + i
            # os.system(cmd_string)
            # print 
            # if file in img_dict:
            #     print "matched"
            #     break
            # print file
            # break
            # print(os.path.join("/mydir", file))

    # print("filter words...")
    # for word, occ in word2occ.items():
    #     if occ >= args.min_occ and word.count('.') <= 1:
    #         word2i[word] = len(word2i)

    # print("Number of words (occ >= 1): {}".format(len(word2occ)))
    # print("Number of words (occ >= {}): {}".format(args.min_occ, len(word2i)))

    # dict_path = os.path.join(args.data_dir, 'dict.json')
    # print("Dump file: {} ...".format(dict_path))
    # with io.open(dict_path, 'wb') as f_out:
    #     data = json.dumps({'word2i': word2i}, ensure_ascii=False)
    #     f_out.write(data.encode('utf8', 'replace'))

    # print("Done!")
