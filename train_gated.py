import sys
sys.path.append('./')
import random
random.seed(42)
from tqdm import tqdm
import os
import pandas as pd
import torch.nn.functional as F
import argparse
import torch
from data import GatedDataset
from torch.utils.data import DataLoader
from model import Prompt_Classifier, Contra_Classifier
from lightning import Fabric
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import yaml
from transformers import CLIPTextModel, CLIPTokenizer
from utils.dataset_utils import load_MyData,prompt_class
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data.dataloader import default_collate
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def collate_fn(batch):
    # 首先使用default_collate处理大部分情况
    prompt,indices,label,id = default_collate(batch)
    encoded_batch = tokenizer.batch_encode_plus(
        prompt,
        return_tensors="pt",
        max_length=77,
        padding='max_length',
        truncation=True,
        )
    return encoded_batch,indices,label,id

def train(opt):
    torch.set_float32_matmul_precision("medium")
    if opt.device_num>1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num,strategy=ddp_strategy)#
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num)
    fabric.launch()

    print(f"Using [{opt.path}] datasets for training")
    dataset=load_MyData(file_folder=opt.path)
    prompt_dataset = GatedDataset(dataset['train'],need_ids=True)
    val_dataset = GatedDataset(dataset['valid'],need_ids=True)

    prompt_dataloader = DataLoader(prompt_dataset, batch_size=opt.per_gpu_batch_size,\
                                     num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
    
    val_dataloder = DataLoader(val_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                            num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    
    if opt.detail_class:
        opt.classifier_dim=len(prompt_class)
        model = Prompt_Classifier(opt,fabric).train()
    elif opt.single_contra:
        opt.classifier_dim=len(prompt_class)
        model = Contra_Classifier(opt,fabric).train()
    else:
        opt.a=opt.b=0
        opt.c=1
        opt.classifier_dim=2
        model = Prompt_Classifier(opt,fabric).train()

    if opt.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            if 'emb' in name:
                param.requires_grad=False
    if opt.c==0:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad=False

    prompt_dataloader,val_dataloder=fabric.setup_dataloaders(prompt_dataloader,val_dataloder)

    if fabric.global_rank == 0 :
        for num in range(10000):
            if os.path.exists(os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num)))==False:
                opt.savedir=os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num))
                os.makedirs(opt.savedir)
                break
        if os.path.exists(os.path.join(opt.savedir,'runs'))==False:
            os.makedirs(os.path.join(opt.savedir,'runs'))
        writer = SummaryWriter(os.path.join(opt.savedir,'runs'))

        #save opt to yaml
        opt_dict = vars(opt)
        with open(os.path.join(opt.savedir,'config.yaml'), 'w') as file:
            yaml.dump(opt_dict, file, sort_keys=False)

    
    num_batches_per_epoch = len(prompt_dataloader)
    warmup_steps=opt.warmup_steps
    lr = opt.lr
    total_steps = opt.total_epoch * num_batches_per_epoch- warmup_steps
    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=lr/10)
    model, optimizer = fabric.setup(model, optimizer)
    max_avg_rec=0


    for epoch in range(opt.total_epoch):
        model.train()
        avg_loss=0
        pbar = enumerate(prompt_dataloader)
        right_num, tot_num= 0,0
        if fabric.global_rank == 0:
            label_dict={}            
            pbar = tqdm(pbar, total=len(prompt_dataloader))
            print(('\n' + '%11s' *(5)) % ('Epoch', 'GPU_mem', 'Cur_loss', 'avg_loss','train_acc'))
        for i,batch in pbar:
            optimizer.zero_grad()
            current_step=epoch*num_batches_per_epoch+i
            if current_step < warmup_steps:
                current_lr = lr * current_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            current_lr = optimizer.param_groups[0]['lr']

            encoded_batch,indices,label,id = batch
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}

            # print(encoded_batch.keys())

            # exit(0)
            if opt.single_contra:
                loss,loss_toxic,loss_benign,loss_classfiy,k,k_label,out,ids  = model(encoded_batch,indices,label,id)
            else:
                loss,loss_nsfw,loss_toxic,loss_benign,loss_classfiy,k,k_label,out,ids= model(encoded_batch,indices,label,id)

            # 对抗训练部分
            if hasattr(opt, 'adv_train') and opt.adv_train:
                from attack_utils import linf_pgd_attack_on_q, l2_pgd_attack_on_q  # 新增
                if opt.adv_type == 'l2_pgd':
                    q_adv = l2_pgd_attack_on_q(model, encoded_batch, indices, label, id, epsilon=3.0, alpha=0.5, num_iter=20)
                elif opt.adv_type == 'linf_pgd':
                    q_adv = linf_pgd_attack_on_q(model, encoded_batch, indices, label, id, epsilon=0.3, alpha=0.1, num_iter=20)
                # 用对抗的q_adv继续后续流程
                k = q_adv.clone().detach()
                k = model.fabric.all_gather(k).view(-1, k.size(1))
                k_label = model.fabric.all_gather(label).view(-1)
                k_ids = model.fabric.all_gather(id).view(-1)
                logits_benign, logits_toxic = model._compute_logits(q_adv, indices, label, k, indices, k_label)
                binary_out = model.classifier(q_adv)
                binary_loss = F.cross_entropy(binary_out, label)
                gt_toxic = torch.zeros(logits_toxic.size(0), dtype=torch.long, device=logits_toxic.device)
                gt_benign = torch.zeros(logits_benign.size(0), dtype=torch.long, device=logits_benign.device)
                if logits_benign.numel() != 0:
                    loss_benign = F.cross_entropy(logits_benign.to(torch.float64), gt_benign)
                else:
                    loss_benign = torch.tensor(0, device=q_adv.device)
                if logits_toxic.numel() != 0:
                    loss_toxic = F.cross_entropy(logits_toxic.to(torch.float64), gt_toxic)
                else:
                    loss_toxic = torch.tensor(0, device=q_adv.device)
                loss = model.a * loss_toxic + model.a * loss_benign + model.c * binary_loss

            preds=torch.argmax(out,dim=1)
            cur_right_num = (preds == k_label).sum().item()
            cur_num = k_label.shape[0]
            right_num+=cur_right_num
            tot_num+=cur_num
            train_acc=right_num/tot_num
            avg_loss=(avg_loss*i+loss.item())/(i+1)
            fabric.backward(loss)
            optimizer.step()
            if current_step >= warmup_steps:
                schedule.step()

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            if fabric.global_rank == 0:
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * 3) %
                    (f'{epoch + 1}/{opt.total_epoch}', mem, loss.item(),avg_loss, train_acc)
                )

                if current_step%10==0:
                    writer.add_scalar('lr', current_lr, current_step)
                    writer.add_scalar('loss', loss.item(), current_step)
                    writer.add_scalar('avg_loss', avg_loss, current_step)
                    writer.add_scalar('loss_classfiy', loss_classfiy.item(), current_step)
                    # writer.add_scalar('loss_nsfw', loss_nsfw.item(), current_step)
                    writer.add_scalar('loss_toxic', loss_toxic.item(), current_step)
                    writer.add_scalar('loss_benign', loss_benign.item(), current_step)
                    writer.add_scalar('train_acc', train_acc, current_step)
        
        with torch.no_grad():
            test_loss=0
            model.eval()
            pbar=enumerate(val_dataloder)
            if fabric.global_rank == 0 :
                test_embeddings,test_labels = [],[]           
                pbar = tqdm(pbar, total=len(val_dataloder))
                print(('\n' + '%11s' *(5)) % ('Epoch', 'GPU_mem', 'Cur_acc', 'avg_acc','loss'))

            right_num, tot_num= 0,0
            for i, batch in pbar:
                encoded_batch,indices,label,id = batch
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                loss,out,k_out,k_outlabel= model(encoded_batch,indices,label,id)
                preds = torch.argmax(out, dim=1)
                # print(preds.shape,k_outlabel.shape)
                cur_right_num = (preds == k_outlabel).sum().item()
                cur_num = k_outlabel.shape[0]

                right_num+=cur_right_num
                tot_num+=cur_num

                test_loss=(test_loss*i+loss.item())/(i+1)

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                if fabric.global_rank == 0 :
                    test_embeddings.append(k_out.cpu())
                    test_labels.extend(k_outlabel.cpu().tolist())
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * 3) %
                        (f'{epoch + 1}/{opt.total_epoch}', mem, cur_right_num/cur_num, right_num/tot_num,loss.item())
                    )
                    
        torch.cuda.empty_cache()
        fabric.barrier()

        if fabric.global_rank == 0:
            writer.add_scalar('val/acc_classifier', right_num/tot_num, epoch)
            avg_rec=right_num/tot_num
            if avg_rec>max_avg_rec:
                max_avg_rec=avg_rec
                torch.save(model.get_encoder().state_dict(), os.path.join(opt.savedir,'model_best.pth'))
                torch.save(model.state_dict(), os.path.join(opt.savedir,'model_classifier_best.pth'))
                print('Save model to {}'.format(os.path.join(opt.savedir,'model_best.pth'.format(epoch))), flush=True)
            
            if epoch%10==0:
                #仅保存编码器，和保存全部参数
                torch.save(model.get_encoder().state_dict(), os.path.join(opt.savedir,'model_{}.pth'.format(epoch)))
                torch.save(model.state_dict(), os.path.join(opt.savedir,'model_classifier_{}.pth'.format(epoch)))
                print('Save model to {}'.format(os.path.join(opt.savedir,'model_{}.pth'.format(epoch))), flush=True)

        fabric.barrier()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=1, help="GPU number to use")
    parser.add_argument('--projection_size', type=int, default=768, help="Pretrained model output dim")
    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers for dataloader")
    parser.add_argument("--per_gpu_batch_size", default=4, type=int, help="Batch size per GPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU for evaluation."
    )

    parser.add_argument("--path", type=str, default="datasets/gated", help="path to dataset")

    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--b', type=float, default=1) 
    parser.add_argument('--c', type=float, default=1)
 
    parser.add_argument('--classifier_dim', type=int, default=2,help="classifier out dim")
    parser.add_argument('--detail_class',action='store_true')
    parser.add_argument('--single_contra',action='store_true')

    parser.add_argument("--total_epoch", type=int, default=10, help="Total number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=300, help="Warmup steps")
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
    parser.add_argument("--eps", type=float, default=1e-6, help="eps")
    parser.add_argument("--savedir", type=str, default="./out")
    parser.add_argument("--name", type=str, default='gated_network')

    parser.add_argument("--resum", type=bool, default=False)
    parser.add_argument("--pth_path", type=str, default='', help="resume embedding model path")

    parser.add_argument("--freeze_embedding_layer",action='store_true',help="freeze embedding layer")

    parser.add_argument("--single_class", action='store_true',help="only use classifier, no contrastive loss")
    parser.add_argument("--tokenizer",type=str,default="/home/raykr/models/CompVis/stable-diffusion-v1-4")
    parser.add_argument("--model_path",type=str,default="/home/raykr/models/CompVis/stable-diffusion-v1-4")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument("--adv_train", action='store_true', help="Enable adversarial training")
    parser.add_argument("--adv_type", type=str, default='linf_pgd', choices=['linf_pgd', 'l2_pgd'], help="Type of adversarial attack")

    opt = parser.parse_args()
    # tokenizer = CLIPTokenizer.from_pretrained(opt.model_path,subfolder='tokenizer')
    tokenizer = CLIPTokenizer.from_pretrained(
    opt.tokenizer,
    subfolder="tokenizer",  
    revision=opt.revision
)

    text_encoder = CLIPTextModel.from_pretrained(
        opt.tokenizer,
        subfolder="text_encoder",  
        revision=opt.revision
    )
    train(opt)



