#import tokenizers
import json
from tqdm import tqdm 
from tokenizers import ByteLevelBPETokenizer 
def Prep_Need_Data(Input_Path,Output_File):
    """
    Single time use function converting the
    json file into a tokenizer library consumable file.
    """
    File = open(Output_File,"w",encoding="utf-8")
    with open(Input_Path) as json_file:
        Data = json.load(json_file)
    for Sample in tqdm(Data):
        NL = " ".join(Sample["NL"])
        Cmd = " ".join(Sample["Cmd"])
        valid_utf8 = True
        String = NL+Cmd
        try:
            String.encode().decode('utf-8')
            File.write(NL+"\n")
            File.write(Cmd+"\n")
        except:
            valid_utf8 = False
            print(String)

    File.close()
    return None

def Tok_Train(input_file_path,vocab_size,output_path):
    """Train a Simple BPE Tokenizer"""
    GPTToken = ByteLevelBPETokenizer(lowercase=False)
    GPTToken.train([input_file_path],vocab_size=3000,min_frequency=2)
    GPTToken.save_model(output_path)
    return None

if __name__ == "__main__":
    Prep_Need_Data(r"G:\Work Related\Nlc2cmd\Data\Template.json",r"G:\Work Related\Nlc2cmd\Tokenizer_Train\Tokenizer Data\Combined.txt")
    Flat_File = r"G:\Work Related\Nlc2cmd\Tokenizer_Train\Tokenizer Data\Combined.txt"
    Tok_Train(Flat_File,3000,r"G:\Work Related\Nlc2cmd\Tokenizer_Train\GPTToken")