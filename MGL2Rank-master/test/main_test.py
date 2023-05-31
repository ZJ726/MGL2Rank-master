import argparse
from sklearn.metrics import f1_score
from time import strftime
import torch.optim as optim
from numpy import *
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
# from model.little import *
from model.Critic_RNNAttn import *
from model.models import *
# from model.onlySiamRank import *
from new_dataloader import *
# from sample_updata import walk_dic_featwalk_new
from sample_updata import walk_dic_featwalk_new

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=3407, help='Random seed.')  # 10
parser.add_argument('--modelName', default="Dir_LSTM_Rank", help='model name')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.') # 太多后面会过拟合，
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')  #0.001
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.45, help='Dropout rate (1 - keep probability).')
parser.add_argument('--clip_gradient', type=float, default=0.2, help='gradient clipping')
parser.add_argument('--d_e', default=16, type=int)
parser.add_argument('--d_h', default=12, type=int)
parser.add_argument('--num_paths', type=int, default=150)
parser.add_argument('--alpha', type=float, default=0.0001)
parser.add_argument('--path_length', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)  # n = 12,32,64
parser.add_argument('--patience', type=int, default=30) # 200
parser.add_argument('--print-freq', default=3000, type=int)
parser.add_argument('--device',default='cuda:0')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed) # 为CPU设置种子

def train(idx_train):
    train_num = idx_train.shape[0] # train_num:697
    splitnum = int(ceil(float(train_num) / args.batch_size))  # splitnum=11
    model.train()

    index = [i for i in range(train_num)]
    random.seed(args.seed)
    random.shuffle(index)
    idx_train_shuf = idx_train[index]

    acclst_micro = []
    acclst_macro = []
    losslst = []
    Difflst = []
    for batch_idx in range(splitnum):
        optimizer.zero_grad()
        indexblock = args.batch_size * batch_idx
        batch_node_idx = idx_train_shuf[range(indexblock, indexblock + min(train_num - indexblock, args.batch_size))]
        sentences = []
        for i in batch_node_idx:
            sentences.extend(sentencedic[i])
        input = featur_corr[sentences]
        # 输入是节点:
        score,idx_1,idx_2 = model(input,batch_node_idx) # score>0,则i>j,反之.score的绝对值大小表示i与j之间重要性的差异程度

        if score is None:
            continue

        label = get_groundtruth_BCE(batch_node_idx, score_mis_order).cuda() # 真实的n(n-1)/2个pairs的两两排名标签,1:i>j,0:i<j)
        loss = criterion(score, label)
        losslst.append(loss.item())
        loss.backward()
        optimizer.step()

        # 评价指标3:基于BCEloss
        pred_res = (score > 0.5).int()
        train_acc_micro = f1_score(label, pred_res, average='micro')
        train_acc_macro = f1_score(label, pred_res, average='macro')
        acclst_micro.append(train_acc_micro)
        acclst_macro.append(train_acc_macro)

        # 评价指标2：
        label_order = getLabelOrder(batch_node_idx)
        preds_order = getPredOrder(idx_1,idx_2,pred_res,batch_node_idx)
        diff = measure(preds_order,label_order,args.batch_size)
        Difflst.append(diff.item())

    return mean(losslst),mean(acclst_micro),mean(acclst_macro),mean(Difflst)

def eval_(idx_input):
    eval_num = idx_input.shape[0]
    splitnum_eval = int(ceil(float(eval_num) / args.batch_size)) # 验证集中的节点要分为19次进行训练(610/32)
    model.eval()

    with torch.no_grad():
        losslst_eval = []
        val_micro = []
        val_macro =[]
        Difflst_eval = []
        for batch_idx in range(splitnum_eval):  # 0~18
            indexblock = args.batch_size * batch_idx
            batch_node_idx_val = idx_input[range(indexblock, indexblock + min(eval_num - indexblock, args.batch_size))]
            sentences = []
            for i in batch_node_idx_val:
                sentences.extend(sentencedic[i])

            input = featur_corr[sentences].cuda()  # (batch_size*500,6)
            score,idx_1,idx_2 = model(input,batch_node_idx_val)

            if score is None:
                continue

            label = get_groundtruth_BCE(batch_node_idx_val, score_mis_order).cuda()  # 真实的n(n-1)/2个pairs的两两排名标签,1:i>j,0:i<j)
            # label = get_groundtruth_BCE_balance(batch_n_i, batch_n_j, score_mis_order).cuda()
            loss = criterion(score, label)
            losslst_eval.append(loss.item())

            pred_res = (score > 0.5).int()
            eval_acc_micro = f1_score(label, pred_res, average='micro')
            # eval_acc_macro = f1_score(label, pred_res, average='macro')
            val_micro.append(eval_acc_micro)

        return mean(losslst_eval), mean(val_micro)#, mean(val_macro), mean(Difflst_eval)

def test(idx_input):
    eval_num = idx_input.shape[0]  # 验证集节点数量
    splitnum_eval = int(ceil(float(eval_num) / args.batch_size)) # 验证集中的节点要分为19次进行训练(610/32)
    model.eval()

    with torch.no_grad():
        losslst_eval = []
        val_micro = []
        val_macro =[]
        Difflst_eval = []
        for batch_idx in range(splitnum_eval):  # 0~18
            indexblock = args.batch_size * batch_idx
            batch_node_idx_val = idx_input[range(indexblock, indexblock + min(eval_num - indexblock, args.batch_size))]
            sentences = []
            for i in batch_node_idx_val:
                sentences.extend(sentencedic[i])

            input = featur_corr[sentences].cuda()  # (batch_size*500,6)
            score,idx_1,idx_2 = model(input,batch_node_idx_val)  # score>0,则i>j,反之。score的绝对值大小表示i与j之间重要性的差异程度

            if score is None:
                continue

            # loss3:BCEloss
            label = get_groundtruth_BCE(batch_node_idx_val, score_mis_order).cuda()  # 真实的n(n-1)/2个pairs的两两排名标签,1:i>j,0:i<j)
            # label = get_groundtruth_BCE_balance(batch_n_i, batch_n_j, score_mis_order).cuda()
            loss = criterion(score, label)
            losslst_eval.append(loss.item())



            pred_res = (score > 0.5).int()
            eval_acc_micro = f1_score(label, pred_res, average='micro')
            eval_acc_macro = f1_score(label, pred_res, average='macro')
            val_micro.append(eval_acc_micro)
            val_macro.append(eval_acc_macro)

            label_order = getLabelOrder(batch_node_idx_val)
            preds_order = getPredOrder(idx_1, idx_2, pred_res,batch_node_idx_val)
            diff = measure(preds_order, label_order, args.batch_size)
            Difflst_eval.append(diff.item())

        return mean(losslst_eval), mean(val_micro), mean(val_macro), mean(Difflst_eval)

if __name__ == "__main__":
    adj, features, score_mis_order, idx_train,idx_test,idx_val, start_time = dataloader_test_929new_npart()
    print(idx_train)
    print(idx_val)
    print(idx_test)

    featur_corr = torch.cat((features, torch.eye(features.shape[1])), 0)
    featur_corr = featur_corr.float()  # (1008,6)

    # 采样
    sentencedic = walk_dic_featwalk_new(adj, features, num_paths=args.num_paths,path_length=args.path_length, alpha=args.alpha).function()

    data = torch.tensor(data.values)
    sentencedic = data.tolist() # 包含1002个子列表的大列表
    setattr(args,'model_time',strftime('%Y_%m_%d_%H_%M_%S', time.localtime((time.time()))))

    model = Dir_LSTM_Rank(args)
    parameter = filter(lambda p: p.requires_grad, model.parameters())  # 从模型参数集合parameters中找到可变的tensor形成新的参数集合
    optimizer = optim.AdamW(parameter, lr=args.lr, weight_decay=args.weight_decay)  # 定义优化器,lr∈(0.001,0.0025)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)# (30, 0.8),(10,0.01)

    # 将模型和数据放到GPU上进行计算
    if args.cuda:
        model.cuda()
        score_mis_order = score_mis_order.clone().detach().cuda()
        featur_corr = featur_corr.cuda()  # (1008,6)
        idx_train = idx_train.cuda() # 节点
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    criterion = nn.BCELoss()
    writer = SummaryWriter(log_dir=r'runs/'+ args.model_time)

    # 开始训练模型
    early_stop = False
    best_dev_acc = 0
    iters_not_improved = 0
    # val_micro = 0
    # test_diff = 0
    patience = args.patience
    a = 0
    for epoch in range(args.epochs):
        print("************************************")
        print("current epochs: '{}'".format(epoch))

        if early_stop:
            print("Early Stopping. Epoch: {}, Best Dev Recall: {}".format(epoch, best_dev_acc))
            break

        train_loss, train_micro,train_macro, train_diff = train(idx_train)
        print("train_loss: {},train_micro:{},train_macro:{},train_diff: {}".format(train_loss, train_micro,train_macro,train_diff))

        # val_loss, val_micro,val_macro,val_diff = eval(idx_test)
        # print("val_loss: {},val_micro:{},val_macro:{},val_diff: {}".format(val_loss, val_micro,val_macro,val_diff))

        # _, test_micro, test_macro, test_diff = eval(idx_test)
        # print("test_micro: '{}',test_macro: '{}',test_diff: '{}'".format(test_micro,test_macro,test_diff))

        # 快速验证
        # train_loss, train_micro = train(idx_train)
        # print("train_loss: {},train_micro:{}".format(train_loss, train_micro))
        val_loss, val_micro= eval_(idx_val)
        print("val_loss: {},val_micro: '{}'".format(val_loss,val_micro))

        # lr衰减
        present_epoch = epoch
        last_epoch = -1
        if present_epoch == args.epochs:
            break
        if present_epoch > last_epoch:  # -1
            print("此时学习率为:{}".format(optimizer.param_groups[0]['lr']))
            scheduler.step()
        last_epoch = present_epoch

        if val_micro >= best_dev_acc:
            best_dev_acc = val_micro
            iters_not_improved = 0
            _, test_micro,test_macro,test_diff = test(idx_test)
            print("test_micro: '{}',test_macro: '{}',test_diff: '{}'".format(test_micro,test_macro,test_diff))
            # 快速测试:
            # _, test_micro = test(idx_test)
            # print("test_micro: '{}'".format(test_micro))
        else:                         # 若此轮的验证集acc开始下降
            iters_not_improved += 1   # 下降轮数+1，(不测试)
            print("No improved: '{}'".format(iters_not_improved)) #
            if iters_not_improved > patience: # 直到11轮没提升，停止训练
                early_stop = True
                break  # 将整个训练全部停止！
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('micro/train', train_micro, epoch)
        writer.add_scalar('macro/train', train_macro, epoch)
        writer.add_scalar('diff/train', train_diff, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('micro/val', val_micro, epoch)
        # writer.add_scalar('macro/val', val_macro, epoch)
        # writer.add_scalar('diff/val', val_diff, epoch)
        a+=1
    writer.close()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))
    print("eval set results:", "accuracy_micro= {:.4f}".format(best_dev_acc))
    print("Test set results:", "test_micro= {:.4f},test_macro= {:.4f},test_diff= {:.4f},".format(test_micro,test_macro,test_diff))
    # print("Test set results:","test_micro= {:.4f}".format(test_micro))

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save({'model_state_dict': model.state_dict(), 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}, f'saved_models/{args.modelName}_{args.model_time}.pth')

    total_params = np.sum(np.fromiter((p.numel() for p in model.parameters() if p.requires_grad), dtype=np.int64))
    print("Total number of trainable parameters in the model: {}".format(total_params))
