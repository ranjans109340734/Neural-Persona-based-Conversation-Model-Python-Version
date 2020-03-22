import torch
import numpy as np
import string
import linecache


class data:
    
    def __init__(self, params):
        self.params=params
        self.EOT=self.params.vocab_target-1 #vocab_target=25010
        self.EOS=self.params.vocab_target-2
        self.beta=self.params.vocab_target-3
    
    # returns a tensor which is reversed verison of input
    def reverse(self, inp):
        length=inp.size(1)
        output=torch.Tensor(1,length)
        for i in range(length):
            output[0][i]=inp[0][length-i-1]
        return output

    # returns tensor, which cosnsits of numbers which are encodings of words (using vocabulary)
    def spl(self, strs):
        splited = strs.split(" ")       #creating list of words
        tensor = torch.Tensor(1,len(splited)).fill_(0)
        count=0
        for i in range(len(splited)):       #looping through number of words in list
            if splited[i]!="":
                tensor[0][count]=int(splited[i])-1      # -1 since dictionary is 0-indexed
                count=count+1
        return tensor
    
    def get_batch(self, Sequences,isSource):
        max_length=-100
        for i in range(len(Sequences)):
            if Sequences[i].size(1)>max_length:
                max_length=Sequences[i].size(1)
        Words=np.ones((len(Sequences),max_length))
        Words.fill(self.params.vocab_dummy)
        Padding=np.zeros((len(Sequences),max_length))
        for i in range(len(Sequences)):
            if isSource:
                Words[i,max_length-Sequences[i].size(1):max_length] = Sequences[i]
                Padding[i,max_length-Sequences[i].size(1):max_length].fill(1)
            else:
                Words[i,:Sequences[i].size(1)] = Sequences[i]
                Padding[i,:Sequences[i].size(1)].fill(1)
        Mask={}
        Left={}
        for i in range(Words.shape[1]):
            Mask[i]=torch.LongTensor((Padding[:,i] == 0).nonzero()[0].tolist())
            Left[i]=torch.LongTensor((Padding[:,i] == 1).nonzero()[0].tolist())
        Words=torch.from_numpy(Words).long()
        Padding=torch.from_numpy(Padding).float()
        return Words,Mask,Left,Padding
    
    def read_train(self, open_train_file, batch_n):
        Y={}
        Source={} 
        Target={}
        End=0;
        SpeakerID="nil"
        AddresseeID="nil"
        for i in range(self.params.batch_size):     #batch_size=256
            line = linecache.getline(open_train_file,batch_n*self.params.batch_size+i+1)
            if line=="":
                End=1
                break
                
            two_strings=line.split("|")     # creates list of 2 strings
            
            addressee_id="nil"
            space=two_strings[0].index(" ")     #int: index of first space in first string
            addressee_line=two_strings[0][space+1:]     #string: capturing only line of first string
            space = two_strings[1].index(" ")
            speaker_id=int(two_strings[1][:space])-1        #capturing speaker ID
            speaker_line=two_strings[1][space+1:]
            
            if type(addressee_id)!=str:
                if type(AddresseeID)==str:
                    AddresseeID=torch.Tensor([addressee_id])
                else:
                    AddresseeID=torch.cat((AddresseeID,torch.Tensor([addressee_id])),0)
                    
            if type(SpeakerID)==str:
                SpeakerID=torch.LongTensor([speaker_id])
            else:
                SpeakerID=torch.cat((SpeakerID,torch.LongTensor([speaker_id])),0)       #why longtensor?
            
            #Source is a dict with tensors as its values, wrt addressee
            if self.params.reverse:
                Source[i]=self.reverse(self.spl(addressee_line.strip()))        #strip() removes white space in the beginning and end, not in middle
            else:
                Source[i]=self.spl(addressee_line.strip())
            
            #Target is a dict with tensors as its values, wrt speaker
            if self.params.reverse_target:
                C=self.reverse(self.spl(speaker_line.strip()))
                Target[i]=torch.cat((torch.Tensor([[self.EOS]]),torch.cat((C,torch.Tensor([[self.EOT]])),1)),1)     #EOT=25009, EOS=25008
            else:
                Target[i]=torch.cat((torch.Tensor([[self.EOS]]),torch.cat((self.spl(speaker_line.strip()),torch.Tensor([[self.EOT]])),1)),1)

        if End==1:
            return End,{},{},{},{},{},{},{},{},{},{},{},{}
        Words_s,Masks_s,Left_s,Padding_s=self.get_batch(Source,True)
        Words_t,Masks_t,Left_t,Padding_t=self.get_batch(Target,False)
        return End,Words_s,Words_t,Masks_s,Masks_t,Left_s,Left_t,Padding_s,Padding_t,Source,Target,SpeakerID,AddresseeID
