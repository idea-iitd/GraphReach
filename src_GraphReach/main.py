from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter

from args import *
from model import *
from utils import *
from dataset import *

import sys
import numpy


numpy.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(edgeitems=sys.maxsize)

if not os.path.isdir('results'):
    os.mkdir('results')
# args
args = make_args()
print(args)
np.random.seed(123)
np.random.seed()
writer_train = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_train')
writer_val = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_val')
writer_test = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_test')


# set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')


for task in ['link', 'link_pair']:
    args.task = task
    if args.dataset=='All':
        if task == 'link':
            datasets_name =['grid', 'communities','ppi','Cora', 'CiteSeer']
        else:
            datasets_name =['communities','completeEmail', 'email', 'protein','Cora', 'CiteSeer']
    else:
        datasets_name = [args.dataset]
    for dataset_name in datasets_name:
        timestart = time.time()
        
        results = []
        results2 = []
        graph_load_time=[]
        maxEpochs=[]
        maxTrainTimes=[]
        maxInfTimes=[]

        if args.weightedRandomWalk:
            rwstr='WeightedRandomWalk'
        else:
            rwstr='RandomWalk'
        if args.normalized:
            rwnormstr='Normalized'
        else:
            rwnormstr='UnNormalized'
        if args.edgelabel:
            eldl='LabeledEdge'
        else:
            eldl=''
        if args.attention:
            attn='withAttention'
        else:
            attn=''
        if args.fastRandomWalk:
            fastRW='Fast'
        else:
            fastRW=''
        if args.attentionAddSelf:
            addself_atten='attentionAddSelf'
        else:
            addself_atten=''


        for repeat in range(args.repeat_num):
            result_val = []
            result_test = []

            result_test2 = []
            max_Epoch=[]
            Train_Time=[]
            Inf_Time=[]
            maxTrainTime=[]
            maxInfTime=[]
            Embedding=[]
            anchorset_ids=[]
            time1 = time.time()


            data_list,bipartite_list,edge_labels,data_dists_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)

            time2 = time.time()
            load_time=time2-time1
            graph_load_time.append(load_time)

            num_features = data_list[0].x.shape[1]
            num_node_classes = None
            num_graph_classes = None
            if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
                num_node_classes = max([data.y.max().item() for data in data_list])+1
            if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
                num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list])+1
            print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
            nodes = [data.num_nodes for data in data_list]
            edges = [data.num_edges for data in data_list]
            print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes)/len(nodes)))
            print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges)/len(edges)))

            args.batch_size = min(args.batch_size, len(data_list))
            print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

            bestvalauc=0
            for i,data in enumerate(data_list):

                anchorset_ids=preselect_anchor(data,bipartite_list[i],data_dists_list[i],args,select_anchors=args.select_anchors, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')

                if args.AdversarialAttack:
                    addTestPairEdges(task, args, data, anchorset_ids, edge_labels)

                data = data.to(device)
                data_list[i] = data

            Train_time_start = time.time()
            # model
            input_dim = num_features
            output_dim = args.output_dim
            model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                        hidden_dim=args.hidden_dim, output_dim=output_dim,
                        feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
            # loss
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            if 'link' in args.task:
                loss_func = nn.BCEWithLogitsLoss()
                out_act = nn.Sigmoid()

            for epoch in range(args.epoch_num):

                torch.cuda.empty_cache()
                




                if epoch==200:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                model.train()
                optimizer.zero_grad()
                shuffle(data_list)
                effective_len = len(data_list)//args.batch_size*len(data_list)

                for id, data in enumerate(data_list[:effective_len]):
                    out = model(data)

                    edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones([data.mask_link_positive_train.shape[1],], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_train.shape[1],], dtype=pred.dtype)
                    label = torch.cat((label_positive,label_negative)).to(device)
                    loss = loss_func(pred, label)

                    # update
                    loss.backward()
                    if id % args.batch_size == args.batch_size-1:
                        if args.batch_size>1:
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad /= args.batch_size
                        optimizer.step()
                        optimizer.zero_grad()

                    del loss


                if epoch % args.epoch_log == 0:
                    # evaluate
                    model.eval()
                

                    loss_train = 0
                    loss_val = 0
                    loss_test = 0
                    correct_train = 0
                    all_train = 0
                    correct_val = 0
                    all_val = 0
                    correct_test = 0
                    all_test = 0
                    auc_train = 0
                    auc_val = 0
                    auc_test = 0
                    emb_norm_min = 0
                    emb_norm_max = 0
                    emb_norm_mean = 0

                    loss_test2 = 0
                    correct_test2 = 0
                    all_test2 = 0
                    auc_test2 = 0

                    for id, data in enumerate(data_list):
                        out = model(data)
                        emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                        emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                        emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()

                        # train
                        edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_train += loss_func(pred, label).cpu().data.numpy()
                        auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        # val
                        edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_val += loss_func(pred, label).cpu().data.numpy()
                        auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        if bestvalauc<auc_val:
                            bestvalauc=auc_val

                            checkpoint = {'model': locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                            hidden_dim=args.hidden_dim, output_dim=output_dim,
                            feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout),
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()}

                            torch.save(checkpoint,'{}_{}_{}_layer{}_{}Anchors_{}_{}{}{}_WalkLength{}_NumWalks{}_p{}_q{}{}_{}_Anchors_{}.pth'.format(args.task,args.model,dataset_name,args.layer_num,args.select_anchors,eldl,fastRW,rwnormstr,rwstr,args.walk_length,args.num_walks,args.p,args.q,attn,addself_atten,args.Num_Anchors))

                        Train_time_end = time.time()

                        Inf_time_start = time.time()
                        # test
                        edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_test += loss_func(pred, label).cpu().data.numpy()
                        auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        Inf_time_end = time.time()

                        # Test 1
                        if args.AdversarialAttack:
                            dists_max_temp = data.dists_max.clone()
                            dists_max_temp2 = data.dists_max2.clone()
                            dists_argmax_temp = data.dists_argmax.clone()

                            data.dists_max = data.dists_max_advAttack.clone()
                            data.dists_max2 = data.dists_max2_advAttack.clone()
                            data.dists_argmax = data.dists_argmaxadvAttack.clone()

                            out = model(data)

                            edge_mask_test2 = np.concatenate((data.mask_link_positive_test2, data.mask_link_negative_test2), axis=-1)
                            nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test2[0, :]).long().to(device))
                            nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test2[1, :]).long().to(device))
                            pred = torch.sum(nodes_first * nodes_second, dim=-1)
                            label_positive = torch.ones([data.mask_link_positive_test2.shape[1], ], dtype=pred.dtype)
                            label_negative = torch.zeros([data.mask_link_negative_test2.shape[1], ], dtype=pred.dtype)
                            label = torch.cat((label_positive, label_negative)).to(device)
                            loss_test2 += loss_func(pred, label).cpu().data.numpy()
                            auc_test2 += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

                            data.dists_max = dists_max_temp.clone()
                            data.dists_max2 = dists_max_temp2.clone()
                            data.dists_argmax = dists_argmax_temp.clone()


                        Training_time=Train_time_end-Train_time_start                        
                        Train_Time.append(Training_time)

                        Inf_time=Inf_time_end-Inf_time_start                        
                        Inf_Time.append(Inf_time)

                    loss_train /= id+1
                    loss_val /= id+1
                    loss_test /= id+1
                    emb_norm_min /= id+1
                    emb_norm_max /= id+1
                    emb_norm_mean /= id+1
                    auc_train /= id+1
                    auc_val /= id+1
                    auc_test /= id+1

                    loss_test2 /= id+1
                    auc_test2 /= id+1

                    if args.AdversarialAttack:
                        print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                          'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test) ,'ADVERSARIAL_ATTACK Test AUC: {:.4f}'.format(auc_test2))
                    else:
                        print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                          'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test) )


                    writer_train.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_train, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_train, epoch)
                    writer_val.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_val, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_val, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_test, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_test, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_min_'+dataset_name, emb_norm_min, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_max_'+dataset_name, emb_norm_max, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_mean_'+dataset_name, emb_norm_mean, epoch)
                    result_val.append(auc_val)
                    result_test.append(auc_test)

                    if args.AdversarialAttack:
                        result_test2.append(auc_test2)


            result_val = np.array(result_val)
            result_test = np.array(result_test)
            results.append(result_test[np.argmax(result_val)])
            max_Epoch=int((np.argmax(result_val)))*int(args.epoch_log)
            maxTrainTime=Train_Time[np.argmax(result_val)]
            maxInfTime=Inf_Time[np.argmax(result_val)]

            if args.AdversarialAttack:
                result_test2 = np.array(result_test2)
                results2.append(result_test2[np.argmax(result_val)])

            print('Max Epoch : ', max_Epoch)
            print('Max Training Time : ',maxTrainTime)
            print('Max Inference Time : ',maxInfTime)
            maxEpochs.append(max_Epoch)
            maxTrainTimes.append(maxTrainTime)
            maxInfTimes.append(maxInfTime)
        results = np.array(results)
        results_mean = np.mean(results).round(3)
        results_std = np.std(results).round(3)
        print('-----------------ROC AUC Scores-------------------')
        print(results_mean, results_std)
        print(results)

        if args.AdversarialAttack:
            results2 = np.array(results2)
            results2_mean = np.mean(results2).round(3)
            results2_std = np.std(results2).round(3)
            print('-----------------AdversarialAttack ROC AUC Scores-------------------')
            print(results2_mean, results2_std)
            print(results2)


        print('*****************************************')
        print('Graph Loading Time : ',graph_load_time)
        print('Max Epochs : ', maxEpochs)
        print('Max Training Times : ',maxTrainTimes)
        print('Max Inference Times : ',maxInfTimes)
        print('*****************************************')

        timeend = time.time()
        exec_time=timeend-timestart
        avg_graph_load_time=np.mean(graph_load_time).round(4)
        avg_maxTrainTimes=np.mean(maxTrainTimes).round(4)
        avg_maxInfTimes=np.mean(maxInfTimes).round(4)
        avg_maxEpochs=np.mean(maxEpochs).round(4)
        print('Execution Time',exec_time)

        print('Avg Graph Loading Time : ', avg_graph_load_time)
        print('Avg of max Training Time : ',avg_maxTrainTimes)
        print('Avg of max Inference Time : ',avg_maxInfTimes)
        print('Avg Max Epochs : ',avg_maxEpochs)

            
        with open('results/{}_{}_{}_layer{}_{}Anchors_{}_{}{}{}_WalkLength{}_NumWalks{}_p{}_q{}{}_{}_Anchors_{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.select_anchors,eldl,fastRW,rwnormstr,rwstr,args.walk_length,args.num_walks,args.p,args.q,attn,addself_atten,args.Num_Anchors), 'w') as f:
            f.write('{}, {}\n'.format(results_mean, results_std))
            f.write('Result : {}\n\n'.format(results))
            
            if args.AdversarialAttack:
                f.write('Adversarial Attack : ')
                f.write('{}, {}\n'.format(results_mean, results_std))
                f.write('Result : {}\n\n'.format(results2))

            f.write('Avg Graph Load Time : {}\n'.format(avg_graph_load_time))
            f.write('Graph Loading Time : {}\n\n'.format(graph_load_time))

            f.write('Program Exection Time : {}\n\n'.format(exec_time))

            f.write('Avg of max Training Time : {}\n'.format(avg_maxTrainTimes))
            f.write('Max Training Times : {}\n\n'.format(maxTrainTimes))    

            f.write('Avg of max Inference Time : {}\n'.format(avg_maxInfTimes))
            f.write('Max Inference Times : {}\n\n'.format(maxInfTimes))

            f.write('Avg Max Epochs : {}\n'.format(avg_maxEpochs))
            f.write('Max Epochs : {}\n'.format(maxEpochs))

writer_train.export_scalars_to_json("./all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json("./all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json("./all_scalars.json")
writer_test.close()


