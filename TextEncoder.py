import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        """
        Create the model and set its weights frozen. 
        Use Transformers library docs to find out how to do this.
        """
        # use the CLS token hidden representation as the sentence's embedding
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel(config)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Pass the arguments through the model and make sure to return CLS token embedding
        """
        output = self.model(input_ids, attention_mask)['last_hidden_state']
        cls_output = output[:, self.target_token_idx, :]
        return cls_output
