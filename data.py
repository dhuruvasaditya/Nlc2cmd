import json
from  torch.utils.data import Dataset 
from pandas import read_json
from tqdm import tqdm
from src.submission_code.encoder_decoder.data_utils import nl_to_partial_tokens,cm_to_partial_tokens
from nlp_tools import tokenizer
from bashlint import data_tools

def Cust_Cmd_Tokenizer(String,parse="Template"):
    '''As per our need Custom CMD Tokenizer'''
    if parse == "Norm":
        Command = cm_to_partial_tokens(String,
                        tokenizer=data_tools.bash_tokenizer)
    elif parse == "Template":
        AST = data_tools.bash_parser(String)
        Template =data_tools.ast2template(AST, ignore_flag_order=False)
    Template_Tokens_List = Template.split(" ")
    return Template_Tokens_List

def Cust_NL_Tokenizer(String,parse="Template"):
    '''Custom NL Tokenizer'''
    if parse == "Template":
        Tokens_List = nl_to_partial_tokens(String,tokenizer=tokenizer.ner_tokenizer)
    elif parse == "Norm":
        Tokens_List = nl_to_partial_tokens(String,tokenizer=tokenizer.basic_tokenizer)
    return Tokens_List


class NlCmdset(Dataset):
    '''Torch Style Dataset for using models'''
    def __init__(self,file_dir,vocab_dir):
        '''
        file_dir : Dataset.json file
        '''
        self.Data = read_json(file_dir)
        self.Load_Vocab(vocab_dir)
        self.Data_Prep(self.Data)
    def Load_Vocab(self,Vocab):
        self.Vocab =  {"i":0}
        return None
    def __len__(self):
        return len(self.Data)
    def Vectorize_Ind(self,Sample,vocab):
        return [vocab["i"] for i in Sample]
    def Data_Prep(self,Data):
        print("Starting Loading Data.")
        NL = []
        Cmd = []
        for key in Data:
            Need = Data[key]
            try:
                Vec_Lang =  self.Vectorize_Ind(Cust_NL_Tokenizer(Need["invocation"]),self.Vocab)
                Vec_Cmd  =  self.Vectorize_Ind(Cust_Cmd_Tokenizer(Need["cmd"]),self.Vocab)
                NL.append(Vec_Lang)
                Cmd.append(Vec_Cmd)
            except:
                pass
        self.NL,self.Cmd = NL,Cmd
        return None
    def __getitem__(self, index):
        Lang = self.NL[index]
        Cmd = self.Cmd[index]      
        return [Lang,Cmd]

if __name__ == "__main__":
    print(Cust_NL_Tokenizer(r'Execute md5sum command on files found by the find command'))
    print(Cust_NL_Tokenizer(r'find all the files in ".\data\utils" '))
    print(Cust_Cmd_Tokenizer(r'find  ".\data\utils"'))
    print(Cust_Cmd_Tokenizer(r'find . -perm -600 -print'))
    TranslationSet = NlCmdset(r"G:\Work Related\NL2Bash\BASH\Data\nl2bash-data.json","..\Data")
    print(TranslationSet[2])