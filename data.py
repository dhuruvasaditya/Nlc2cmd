import json
from  torch.utils.data import Dataset 
from pandas import read_json
from tqdm import tqdm
from src.submission_code.encoder_decoder.data_utils import nl_to_partial_tokens,cm_to_partial_tokens
from nlp_tools import tokenizer
from bashlint import data_tools
from torch import stack,Tensor

def Cust_Cmd_Tokenizer(String,parse="Template"):
    """As per our need Custom CMD Tokenizer"""
    if parse == "Norm":
        Command = cm_to_partial_tokens(String,
                        tokenizer=data_tools.bash_tokenizer)
    elif parse == "Template":
        AST = data_tools.bash_parser(String)
        Template =data_tools.ast2template(AST, ignore_flag_order=False)
    Template_Tokens_List = Template.split(" ")
    return Template_Tokens_List

def Cust_NL_Tokenizer(String,parse="Template"):
    """Custom NL Tokenizer"""
    if parse == "Template":
        Tokens_List = nl_to_partial_tokens(String,tokenizer=tokenizer.ner_tokenizer)
    elif parse == "Norm":
        Tokens_List = nl_to_partial_tokens(String,tokenizer=tokenizer.basic_tokenizer)
    return Tokens_List

class WriteJSON(Dataset):
    """Torch Style Iterator Dataset for use"""
    def __init__(self,file_dir,vocab_dir):
        """
        file_dir : Dataset.json file
        """
        self.Data = read_json(file_dir)
        self.Data_Prep(self.Data)
        self.write_json(self.NL_Cmd,r"G:\Work Related\Nlc2cmd\Data\Template.json")
    def write_json(self,NL_CMD,path):
        Dict = NL_CMD
        with open(path, 'w') as outfile:
            json.dump(Dict, outfile)
    def Vectorize_Ind(self,Sample,vocab):
        return [vocab["i"] for i in Sample]
    def Data_Prep(self,Data):
        print("Starting Loading Data.")
        NL_Cmd = []
        for key in Data:
            Need = Data[key]
            try:
                Vec_Lang = Cust_NL_Tokenizer(Need["invocation"]) 
                Vec_Cmd  = Cust_Cmd_Tokenizer(Need["cmd"])
                Need = {"NL":Vec_Lang,"Cmd":Vec_Cmd}
                NL_Cmd.append(Need)
            except:
                pass
        self.NL_Cmd = NL_Cmd
        return None

def pad_and_get_mask(mode,code, nl, tokenizer,block_size):
    '''pad_and_get_mask function block'''
    if mode == 'test':
        code = []
    while (len(code) + len(nl) + 2 > block_size):
        if (len(code) > len(nl)):
            code = code[:-1]
        else:
            nl = nl[:-1]
    if mode == 'train':
        inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
        labels = [1] * len(nl) + [2] * (len(code)+1) + [0]
    else:
        inputs = nl + [tokenizer.bos_token_id]
        labels = [1] * len(nl) + [2]
        return inputs, labels
    assert len(inputs) <= block_size
    pad_len = block_size - len(inputs)
    inputs += [tokenizer.pad_token_id] * pad_len
    labels += [0] * pad_len
    assert len(inputs) == len(labels)
    return inputs, labels


class OmnibashDataset(Dataset):
    '''OmniBASH Dataset torch.utils.data.Dataset Object
    Data_Path = Json Data with NL and Command.
    Tokenizer = Bash Dataset tokenizers Object
    Mode      = Train/Test which preprocesses and makes Attention Mask.
    Block_Size = transformers.dataset Block Size*(Pad Length of NL and Cmd) param.
    '''
    def __init__(self,Data_Path,Tokenizer,Mode,Block_Size):
        self.Data = read_json(Data_Path)
        self.Tokenizer = Tokenizer
        self.mode = Mode
        self.Block_Size = Block_Size
    def Load_Json_Data_Template(self,Json):
        Input_Set = []
        Input_Label = []
        for data in Json:
            NL = " ".join(data["NL"])
            Cmd =" ".join(data["Cmd"])
            Input,Label = pad_and_get_mask(self.mode,Cmd,NL,self.Tokenizer,self.Block_Size)
            self.Input_Set.append(Input)
            self.Input_Label.append(Label)
        Input = stack(Input_Set)
        Labels = stack(Input_Label)
        self.Inputs,self.Labels = Input,Labels
        return  None
    def __getitem__(self, index):
        return self.Inputs[index],self.Labels[index]
    def __len__(self):
        return len(self.Data)


if __name__ == "__main__":
    OmnibashDataset(r"G:\Work Related\Nlc2cmd\Data\Template.json")
    #print(Cust_NL_Tokenizer(r'Execute md5sum command on files found by the find command'))
    #print(Cust_NL_Tokenizer(r'find all the files in ".\data\utils" '))
    #print(Cust_Cmd_Tokenizer(r'find  ".\data\utils"'))
    #print(str(Cust_Cmd_Tokenizer(r'find . -perm -600 -print')))
    #WriteJSON(r"G:\Work Related\Nlc2cmd\Data\nl2bash-data.json"
    #                            ,"..\Data")
    
