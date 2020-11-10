import json
from tqdm import tqdm

def Read_Write(Json_Path,Output_Path):
    File = open(Output_Path,"w",encoding="utf-8")
    with open(Json_Path) as json_file:
        Data = json.load(json_file)
    for Sample in tqdm(Data):
        Cmd = " ".join(Sample["Cmd"])
        valid_utf8 = True
        try:
            Cmd.encode().decode('utf-8')
            File.write(Cmd+"\n")
        except:
            pass
    print("Written Json")
    return None
if __name__ == "__main__":
    Read_Write(r"G:\Work Related\Nlc2cmd\Data\Template.json",r"G:\Work Related\Nlc2cmd\pretrain_gpt\pretrain_data\Cmd.txt")