from tokenizers import ByteLevelBPETokenizer 


def Tok_Train(input_file_path,vocab_size,output_path):
    """Train a Simple BPE Tokenizer"""
    GPTToken = ByteLevelBPETokenizer(lowercase=True)
    GPTToken.enable_padding()
    GPTToken.train([input_file_path],vocab_size=vocab_size,min_frequency=2,special_tokens=["PAD"])
    GPTToken.save_model(output_path)
    return None

if __name__ == "__main__":
    Tok_Train(r"G:\Work Related\Nlc2cmd\pretrain_gpt\pretrain_data\Cmd.txt",300,r"G:\Work Related\Nlc2cmd\pretrain_gpt\bashgpt")