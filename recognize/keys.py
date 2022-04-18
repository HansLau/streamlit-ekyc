import pickle as pkl
import os
# gen alphabet via label
# alphabet_set = set()
# infofiles = ['infofiles/infofile_selfcollect.txt','infofiles/infofile_train_public.txt']
# for infofile in infofiles:
#     f = open(infofile)
#     content = f.readlines()
#     f.close()
#     for line in content:
#         if len(line.strip())>0:
#             if len(line.strip().split('\t'))!=2:
#                 print(line)
#             else:
#                 fname,label = line.strip().split('\t')
#                 for ch in label:
#                     alphabet_set.add(ch)
#
# alphabet_list = sorted(list(alphabet_set))
# pkl.dump(alphabet_list,open('alphabet.pkl','wb'))

dir_path = os.path.dirname(os.path.realpath(__file__))

alphabet_list = pkl.load(open(dir_path + '/alphabet.pkl','rb'))
alphabet = [ord(ch) for ch in alphabet_list]
alphabet_v2 = alphabet
# print(alphabet_v2)