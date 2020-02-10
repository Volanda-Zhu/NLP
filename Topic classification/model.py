import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, num_class, out_channels, kernel_size, stride, padding, keep_prob, 
        vocab_size, embed_len, weights, fine_tune = False):
        super(CNN, self).__init__()
        

        self.num_class = num_class
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embed_len = embed_len
        
        self.fixed_embed = nn.Embedding(vocab_size, embed_len)
        self.tuned_embed = nn.Embedding(vocab_size, embed_len)

        self.fixed_embed.weight = nn.Parameter(torch.FloatTensor(weights), requires_grad=False)
        self.tuned_embed.weight = nn.Parameter(torch.FloatTensor(weights), requires_grad=True)
        
        self.conv1 = nn.Conv2d(1, out_channels, (kernel_size[0], embed_len), stride, padding)
        self.conv2 = nn.Conv2d(1, out_channels, (kernel_size[1], embed_len), stride, padding)
        self.conv3 = nn.Conv2d(1, out_channels, (kernel_size[2], embed_len), stride, padding)
        self.dropout = nn.Dropout(keep_prob)
        self.label = nn.Linear(len(kernel_size)*out_channels, num_class)
    
    def max_pool(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        
        return max_out
    
    def forward(self, input_sentences, batch_size=None):
    
        
        input_f = self.fixed_embed(input_sentences)
        input_t = self.tuned_embed(input_sentences)
        combined_inputs = torch.add(input_f.unsqueeze(1), input_t.unsqueeze(1))

        
        out1 = self.max_pool(combined_inputs, self.conv1)
        out2 = self.max_pool(combined_inputs, self.conv2)       
        out3 = self.max_pool(combined_inputs, self.conv3)
        
        all_out = torch.cat((out1, out2, out3), 1)
        
        fc = self.dropout(all_out)       
        
        res = self.label(fc)
        
        return res