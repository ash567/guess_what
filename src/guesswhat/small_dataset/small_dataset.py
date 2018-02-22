"""Create a smaller dataset (images + games) from specified GuessWhat dataset

example
-------
python src/guesswhat/preprocess_data/small_dataset.py -data_dir=/path/to/guesswhat/ -no_examples=number of examples

eg:

python /home/ishu/guesswhat/src/guesswhat/small_dataset/small_dataset.py -data_dir /home/ishu/guesswhat/data -no_examples 16

"""

import argparse
import collections
import io
import json
import os
import re
from glob import glob


from guesswhat.data_provider.guesswhat_dataset import OracleDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating Small Dataset')
    parser.add_argument("-data_dir", type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-no_examples", type=int, default=333, help="Number of examples")

    # #
    # data_dir =  "/home/ishu/guesswhat/data"
    # no_examples = 300

    args = parser.parse_args()
    options = [(2, "train"), (1, "test"), (1, "valid")]
    path = args.data_dir
    cwd = os.getcwd()
    
    # for (n, s) in options:
    #     print
    #     print s
    #     print "****"

    #     s = path + "guesswhat." + s
    #     s_big = s + "_big"
    #     gw_data = s + ".jsonl"
    #     com_gw_data = gw_data + ".gz"
    #     big_gw_data = s_big + ".jsonl"
    #     com_big_gw_data =  big_gw_data + ".gz"

    #     cmd = []

    #     if os.path.isfile(com_big_gw_data) and os.path.isfile(com_gw_data):
    #         print "Small dataset already present. Deleting the file already present and creating new files"
    #         print

    #         # Unzipping the dataset
    #         cmd.append("gunzip " + com_big_gw_data)
    #         # Deleting the old file
    #         cmd.append("rm " + com_gw_data)

    #     else:
    #         # Unzipping the dataset
    #         cmd.append("gunzip " + com_gw_data)
    #         # renaming the true large dataset
    #         cmd.append("mv " + gw_data + " " + big_gw_data)

    #     # Making a new small dataset (Taking only some of the games)
    #     cmd.append("head -%d " % (n * args.no_examples) + big_gw_data + " > " + gw_data)
    #     # Zipping back the files
    #     cmd.append("gzip " + big_gw_data)
    #     cmd.append("gzip " + gw_data)

    #     for c in cmd:
    #         print c
    #         os.system(c)
    #         # break
    #     print
    #     print  "----------------"


    print "Printing the image files"
    # game_img_dict = {}
    img_dict = {}
    for (n, s) in options:
        game_set = OracleDataset.load(args.data_dir, s)
        for game in game_set.games:
            img_dict[str(game.image.id) + ".jpg"] = 1
        f = open(path + s + "_img_index.txt", "w")
        for key in img_dict:
            print >>f, key
        f.close()
        # game_img_dict[s] = img_dict
        # print s
        # print img_dict
        print

    print img_dict

    # path of the data
    raw_dir = path + "img/raw"
    big_raw_dir = raw_dir + "_big/"
    raw_dir = raw_dir + "/"
    image_subdir = ["train2014/", "val2014/"]

    # Rename the raw directory if raw big does not exitst
    # Also assuming train2014 and val2014 are present in the system

    if not os.path.exists(big_raw_dir):
        os.system("mv " + raw_dir + " " + big_raw_dir)
    else:
        print "Not deleting the raw_big directory (risky if the arguments go wrong). Delete it yourself"

    if not os.path.exists(raw_dir):
        os.system("mkdir " + raw_dir)

    for directory in image_subdir:    
        if not os.path.exists(raw_dir + directory):
            os.system("mkdir " + raw_dir + directory)

    l = {}
    pattern = re.compile(r'.*_0*(\d+\.\w+)$')

    for directory in image_subdir:
        ids = {}
        for name in glob(os.path.join(big_raw_dir, directory, "*")):
            # The name starts with the absolute name
            res = pattern.match(name)
            if not res:
                continue
            ids[res.group(1)] =  name
        l[directory] = ids

    count = 0
    for directory in l:
        for image_id in l[directory]:
            if image_id in img_dict:
                count = count + 1
                print image_id
                # move the file l[directory][image_id] to raw + directory
                cmd = "cp " + l[directory][image_id] + " " + raw_dir + directory
                print cmd
                os.system(cmd)
        # break
    print count == len(img_dict)
    # Assuming val and train are already present in big_raw


    # cmd_string = []
    # i = 0
    # for file in os.listdir(args.data_dir + "/img/raw"):
    #     if file.endswith(".jpg"):
    #         if file in img_dict:
    #             i = i + 1
    #             continue
    #         # Moving files that are not in the game to other folder
    #         cmd_string.append("mv " + args.data_dir + "/img/raw/" + file + " " + args.data_dir + "/img/raw_big/" + file)
    #         # print cmd_string

    # print len(cmd_string) + i
    #         # os.system(cmd_string)
    #         # print 
    #         # if file in img_dict:
    #         #     print "matched"
    #         #     break
    #         # print file
    #         # break
    #         # print(os.path.join("/mydir", file))

    # # print("filter words...")
    # # for word, occ in word2occ.items():
    # #     if occ >= args.min_occ and word.count('.') <= 1:
    # #         word2i[word] = len(word2i)

    # # print("Number of words (occ >= 1): {}".format(len(word2occ)))
    # # print("Number of words (occ >= {}): {}".format(args.min_occ, len(word2i)))

    # # dict_path = os.path.join(args.data_dir, 'dict.json')
    # # print("Dump file: {} ...".format(dict_path))
    # # with io.open(dict_path, 'wb') as f_out:
    # #     data = json.dumps({'word2i': word2i}, ensure_ascii=False)
    # #     f_out.write(data.encode('utf8', 'replace'))

    # # print("Done!")



# Just storing the id names.
# ----------------------------------------------------------------------

# To check if no two ids are same (sort ids as list and then check using icrement method)
# for i in xrange(0, len(l)):
#     for j in xrange(i + 1, len(l)):
#         print "Inside %d, %d" %(i, j)
#         n = m = 0
#         while(m < len(l[i]) and n < len(l[j])):
#             if l[i][m] < l[j][n]:
#                 m = m + 1
#             elif l[i][m] > l[j][n]:
#                 n = n + 1
#             else:
#                 print "Same at %d, %d" %(i, n)
#                 n = n + 1



                # if l[i][m] == l[j][n]:
                #     print l[i][m]
