#import tokenizers
import json
from tqdm import tqdm 
def Prep_Need_Data(Input_Path,Output_File):
    """
    Single time use function converting the
    json file into a tokenizer library consumable file.
    """
    File = open(Output_File,"w")
    with open(Input_Path) as json_file:
        Data = json.load(json_file)
    for Sample in tqdm(Data):
        NL = " ".join(Sample["NL"])
        Cmd = " ".join(Sample["Cmd"])
        File.write(NL+"\n")
        File.write(Cmd+"\n")
    File.close()
    return None
if __name__ == "__main__":
    Prep_Need_Data(r"G:\Work Related\Nlc2cmd\Data\Template.json",r"G:\Work Related\Nlc2cmd\Tokenizer_Train\Tokenizer Data\Combined.txt")