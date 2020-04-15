# coding: utf-8
import codecs
import sys
import pickle

class pokemon():
    def __init__(self, name, type, h, a, b, c, d, s):
        self.name = name
        self.type = type
        self.h = h
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.s = s

def load_jp_en_type_dict(filename):
    dict = {}
    char_dict = {" ":0}
    char_id_dict = {0:" "}
    with codecs.open(filename,"r","utf-8") as f:
        lines = f.readlines()
        for i,l in enumerate(lines):
            if i==0:
                continue
            l_list = l.split(",")
            en = l_list[2].lower()
            jp = [0] * 6
            for index, c in enumerate(l_list[1]):
                if c not in char_dict:
                    id = len(char_dict)
                    char_dict[c] = id
                    char_id_dict[id] = c
                jp[index]=char_dict[c]
            if en not in dict or len(dict[en][0]) > len(jp):
                if '"' not in l_list[4]:
                    dict[en] = (jp, l_list[4].strip('"'))
                else:
                    dict[en] = (jp, l_list[4].strip('"') + "," + l_list[5].strip('"'))
    return dict, char_dict, char_id_dict

def load_pokemon_dict(jp_en_type_dict, filename):
    dict = {}
    type_dict = {}
    type_id_dict = {}
    with codecs.open(filename,"r","utf-8") as f:
        lines = f.readlines()
        for i,l in enumerate(lines):
            if i==0:
                continue
            l_list = l.split(",")
            en_name = l_list[1].lower()
            if en_name in jp_en_type_dict:
                jp_name = jp_en_type_dict[en_name][0]
                type = [0] * 24
                for t in jp_en_type_dict[en_name][1].split(","):
                    if t not in type_dict:
                        id = len(type_dict)
                        type_dict[t] = id
                        type_id_dict[id] = t
                    type[type_dict[t]] = 1
                type[18] = int(l_list[4])
                type[19] = int(l_list[5])
                type[20] = int(l_list[6])
                type[21] = int(l_list[7])
                type[22] = int(l_list[8])
                type[23] = int(l_list[9])
                if en_name not in dict:
                    dict[en_name] = (jp_name, type)
    return dict, type_dict, type_id_dict



if __name__ == "__main__":
    jp_en_type_dict, char_dict, char_id_dict = load_jp_en_type_dict("pokemon_20191125.csv")
    pokemon_dict, type_dict, type_id_dict = load_pokemon_dict(jp_en_type_dict, "pokemon-with-stats-generation-8/Pokemon_Gen_1-8.csv")
    with open("pokemondata.pkl","wb") as f:
        pickle.dump((pokemon_dict, char_dict, char_id_dict, type_dict, type_id_dict) , f)
    print(len(char_dict))
    print(len(pokemon_dict))