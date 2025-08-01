import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, in_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.dense1 = nn.Linear(in_dim, in_dim//4)
        self.dense2 = nn.Linear(in_dim//4, in_dim//16)
        self.out_proj = nn.Linear(in_dim//16, out_dim)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.dense1.bias, std=1e-6)
        nn.init.normal_(self.dense2.bias, std=1e-6)
        nn.init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, features):
        x = features
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class Prompt_Classifier(nn.Module):
    def __init__(self, opt,fabric):
        super(Prompt_Classifier, self).__init__()

        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric

        self.model=CLIPTextModel.from_pretrained(opt.model_path,subfolder='text_encoder',revision=opt.revision)
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.model.device)
            self.model.load_state_dict(state_dict)
  
        self.device=self.model.device
        self.esp=torch.tensor(1e-6,device=self.device)
        self.a = torch.tensor(opt.a,device=self.device)
        self.b = torch.tensor(opt.b,device=self.device)
        self.c = torch.tensor(opt.c,device=self.device)

        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)


    def get_encoder(self):
        return self.model

    def _compute_logits(self, q,q_index,q_label,k,k_index,k_label):
        def cosine_similarity_matrix(q, k):

            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        

        logits=cosine_similarity_matrix(q,k)/self.temperature
        q_index=q_index.view(-1, 1)# N,1
        q_labels=q_label.view(-1, 1)# N,1

        k_index=k_index.view(1, -1)# 1,N+K
        k_labels=k_label.view(1, -1)# 1,N+K

        same_nsfw=(q_index==k_index)
        same_label=(q_labels==k_labels)# N,N+K

        is_benign=(q_label==1).view(-1)
        is_toxic=(q_label==0).view(-1)


        pos_logits_benign = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_benign=logits*torch.logical_not(same_label)
        logits_benign=torch.cat((pos_logits_benign.unsqueeze(1), neg_logits_benign), dim=1)
        logits_benign=logits_benign[is_benign]

        pos_logits_toxic = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_toxic=logits*torch.logical_not(same_label)
        logits_toxic=torch.cat((pos_logits_toxic.unsqueeze(1), neg_logits_toxic), dim=1)
        logits_toxic=logits_toxic[is_toxic]

        pos_logits_nsfw = torch.sum(logits*same_nsfw,dim=1)/torch.max(torch.sum(same_nsfw,dim=1),self.esp)# N
        neg_logits_nsfw=logits*torch.logical_not(same_nsfw)# N,N+K
        logits_nsfw=torch.cat((pos_logits_nsfw.unsqueeze(1), neg_logits_nsfw), dim=1)
        logits_nsfw=logits_nsfw[is_toxic]

        return logits_benign,logits_nsfw,logits_toxic
    
    def forward(self, encoded_batch, indices,label,id):
        q=self.model(encoded_batch['input_ids'],encoded_batch['attention_mask'])
        q=q.pooler_output

        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index = self.fabric.all_gather(indices).view(-1)
        k_ids=self.fabric.all_gather(id).view(-1)
        
        logits_benign,logits_nsfw,logits_toxic = self._compute_logits(q,indices,label,k,k_index,k_label)
        #查询的经过分类头得到分类
        out = self.classifier(q)
        if self.opt.detail_class:
          loss_classfiy=F.cross_entropy(out,indices)  
        else:  
          loss_classfiy = F.cross_entropy(out, label)

        gt_nsfw = torch.zeros(logits_nsfw.size(0), dtype=torch.long,device=logits_nsfw.device)
        gt_toxic = torch.zeros(logits_toxic.size(0), dtype=torch.long,device=logits_toxic.device)
        gt_benign = torch.zeros(logits_benign.size(0), dtype=torch.long,device=logits_benign.device)

        loss_nsfw = F.cross_entropy(logits_nsfw, gt_nsfw)
        loss_toxic = F.cross_entropy(logits_toxic, gt_toxic)
        loss_benign =  F.cross_entropy(logits_benign, gt_benign)
        if logits_benign.numel()!=0:
            loss_benign = F.cross_entropy(logits_benign.to(torch.float64), gt_benign)
        else:
            loss_benign=torch.tensor(0,device=self.device)

        loss = self.a*loss_nsfw + self.b*loss_toxic +(self.a+self.b)*loss_benign+self.c*loss_classfiy
        out = self.fabric.all_gather(out).view(-1, out.size(1))
        if self.training:
            if self.opt.detail_class:
                return loss,loss_nsfw,loss_toxic,loss_classfiy,loss_benign,k,k_index,out,k_ids
            else:
              return loss,loss_nsfw,loss_toxic,loss_classfiy,loss_benign,k,k_label,out,k_ids
        else:
            if self.opt.detail_class:
                return loss,out,k,k_index
            else:
              return loss,out,k,k_label
        

class Contra_Classifier(nn.Module):
    def __init__(self, opt,fabric):
        super(Contra_Classifier, self).__init__()

        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric

        self.model=CLIPTextModel.from_pretrained(opt.model_path,subfolder='text_encoder',revision=opt.revision)
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.model.device)
            self.model.load_state_dict(state_dict)
  
        self.device=self.model.device
        self.esp=torch.tensor(1e-6,device=self.device)
        self.a = torch.tensor(opt.a,device=self.device)
        self.b = torch.tensor(opt.b,device=self.device)
        self.c = torch.tensor(opt.c,device=self.device)

        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)

    def get_encoder(self):
        return self.model

    def _compute_logits(self, q,q_index,q_label,k,k_index,k_label):
        def cosine_similarity_matrix(q, k):

            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        

        logits=cosine_similarity_matrix(q,k)/self.temperature
        q_labels=q_label.view(-1, 1)# N,1
        k_labels=k_label.view(1, -1)# 1,N+K

        same_label=(q_labels==k_labels)# N,N+K

        is_toxic=(q_label==0).view(-1)
        is_benign=(q_label==1).view(-1)

        pos_logits_benign = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_benign=logits*torch.logical_not(same_label)
        logits_benign=torch.cat((pos_logits_benign.unsqueeze(1), neg_logits_benign), dim=1)
        logits_benign=logits_benign[is_benign]

        pos_logits_toxic = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_toxic=logits*torch.logical_not(same_label)
        logits_toxic=torch.cat((pos_logits_toxic.unsqueeze(1), neg_logits_toxic), dim=1)
        logits_toxic=logits_toxic[is_toxic]

        return logits_benign,logits_toxic
    
    def forward(self, encoded_batch, indices,label,id):
        q=self.model(encoded_batch['input_ids'],encoded_batch['attention_mask'])
        q=q.pooler_output

        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_ids=self.fabric.all_gather(id).view(-1)

        logits_benign,logits_toxic = self._compute_logits(q,indices,label,k,indices,k_label)
        binary_out=self.classifier(q)
        binary_loss=F.cross_entropy(binary_out,label)
        gt_toxic = torch.zeros(logits_toxic.size(0), dtype=torch.long,device=logits_toxic.device)
        gt_benign = torch.zeros(logits_benign.size(0), dtype=torch.long,device=logits_benign.device)
        if logits_benign.numel()!=0:
            loss_benign = F.cross_entropy(logits_benign.to(torch.float64), gt_benign)
        else:
            loss_benign=torch.tensor(0,device=self.device)
        if logits_toxic.numel()!=0:
            loss_toxic = F.cross_entropy(logits_toxic.to(torch.float64), gt_toxic)
        else:
            loss_toxic=torch.tensor(0,device=self.device)
        loss = self.a*loss_toxic + self.a*loss_benign + self.c*binary_loss
        if self.training:
            return loss,loss_toxic,loss_benign,binary_loss,k,k_label,binary_out,k_ids
        else:
            return loss,binary_out,k,k_label
        
class Binary_Classifier(nn.Module):
    """二分类器：benign vs toxic，仅使用交叉熵损失"""
    
    def __init__(self, opt, fabric):
        super(Binary_Classifier, self).__init__()
        
        self.opt = opt
        self.fabric = fabric
        
        self.model = CLIPTextModel.from_pretrained(opt.model_path, subfolder='text_encoder', revision=opt.revision)
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.model.device)
            self.model.load_state_dict(state_dict)
            
        self.device = self.model.device
        
        self.binary_classifier = ClassificationHead(opt.projection_size, 2)
        
    def get_encoder(self):
        return self.model
    
    def forward(self, encoded_batch, indices, label, id):
        # 获取文本特征
        q = self.model(encoded_batch['input_ids'], encoded_batch['attention_mask'])
        q = q.pooler_output
             
        # 二分类预测
        binary_out = self.binary_classifier(q)
        
        # 只使用交叉熵损失
        loss = F.cross_entropy(binary_out, label)
        
        binary_out = self.fabric.all_gather(binary_out).view(-1, binary_out.size(1))
        binary_label = self.fabric.all_gather(label).view(-1)
        k_ids = self.fabric.all_gather(id).view(-1)
        
        if self.training:
            return loss, binary_out, binary_label, k_ids
        else:
            return loss, binary_out, binary_label

    def forward_with_embedding(self, embedding, indices, label, id):
        q = embedding
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index = self.fabric.all_gather(indices).view(-1)
        k_ids = self.fabric.all_gather(id).view(-1)
        logits_benign, logits_nsfw, logits_toxic = self._compute_logits(q, indices, label, k, k_index, k_label)
        out = self.classifier(q)
        if self.opt.detail_class:
            loss_classfiy = F.cross_entropy(out, indices)
        else:
            loss_classfiy = F.cross_entropy(out, label)
        gt_nsfw = torch.zeros(logits_nsfw.size(0), dtype=torch.long, device=logits_nsfw.device)
        gt_toxic = torch.zeros(logits_toxic.size(0), dtype=torch.long, device=logits_toxic.device)
        gt_benign = torch.zeros(logits_benign.size(0), dtype=torch.long, device=logits_benign.device)
        loss_nsfw = F.cross_entropy(logits_nsfw, gt_nsfw)
        loss_toxic = F.cross_entropy(logits_toxic, gt_toxic)
        loss_benign = F.cross_entropy(logits_benign.to(torch.float64), gt_benign) if logits_benign.numel() != 0 else torch.tensor(0, device=self.device)
        loss = self.a * loss_nsfw + self.b * loss_toxic + (self.a + self.b) * loss_benign + self.c * loss_classfiy
        out = self.fabric.all_gather(out).view(-1, out.size(1))
        if self.training:
            if self.opt.detail_class:
                return loss, loss_nsfw, loss_toxic, loss_classfiy, loss_benign, k, k_index, out, k_ids
            else:
                return loss, loss_nsfw, loss_toxic, loss_classfiy, loss_benign, k, k_label, out, k_ids
        else:
            if self.opt.detail_class:
                return loss, out, k, k_index
            else:
                return loss, out, k, k_label