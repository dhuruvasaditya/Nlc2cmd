from pytorch_lightning import LightningModule,Trainer,TrainResult,seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from model import OmniBash
from transformers import OpenAIGPTLMHeadModel,OpenAIGPTConfig,GPT2TokenizerFast
from data import OmnibashDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import stack
class GPTTrain(LightningModule):
    def __init__(self,hparams):
        super(GPTTrain,self).__init__()
        Trans_Config = OpenAIGPTConfig(vocab_size=hparams["vocab_size"],n_layer=hparams["layers_num"])
        Trans_Model = OpenAIGPTLMHeadModel(Trans_Config)
        self.Transformer = OmniBash(Trans_Model,hparams["device"])
        self.Trans_Tok = GPT2TokenizerFast.from_pretrained(hparams["tokenizer_dir"])
        self.hparams = hparams
    def forward(self,Input,Label):
        Logits,Lables = self.Transformer(Input,Label)
        return Logits,Lables
    def configure_optimizers(self):
        return Adam(self.parameters(),lr=self.hparams["lr"])
    def prepare_data(self):
        Dataset = OmnibashDataset(self.hparams["data_path"],self.Trans_Tok,self.hparams["state"],self.hparams["seqlen"])
        self.Dataset = Dataset
    def train_dataloader(self):
        Trainloader = DataLoader(self.Dataset,batch_size=self.hparams["batch_size"])
        return Trainloader
    def Loss_Func(self,Logits,Labels):
        return CrossEntropyLoss()(Logits,Labels)
    def training_step(self,batch,batch_idx):
        Input,Labels  = batch
        Logits,Labels = self.forward(Input,Labels)
        Loss = CrossEntropyLoss()(Logits,Labels)
        Step_Log = {"loss":Loss}
        return Step_Log
    def training_epoch_end(self, outputs):
        Avg_Loss = stack([x['loss'] for x in outputs]).mean()
        Epoch_Log = {"Avg_Loss":Avg_Loss.item()}
        self.logger.experiment.log_metrics(Epoch_Log)
        return Epoch_Log
if __name__ == "__main__":
    import json
    with open('hparams.json') as json_file:
        hparams = json.load(json_file)
    Model = GPTTrain(hparams)
    seed_everything(42)
    Logger = CSVLogger("GPT_Train_Logs",name="Init_Trial",version="1")
    trainer = Trainer(logger=Logger,max_epochs=hparams["max_epoch"])
    trainer.fit(Model)
    Logger.save()
