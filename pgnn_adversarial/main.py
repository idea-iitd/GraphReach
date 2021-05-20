from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter

from args import *
from model import *
from utils import *
from dataset import *

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
            datasets_name = []#['grid','communities','ppi']
        else:
            datasets_name = ['completeEmail']#,'completeEmail', 'email', 'protein']
    else:
        datasets_name = [args.dataset]
    for dataset_name in datasets_name:
        # if dataset_name in ['communities','grid']:
        #     args.cache = False
        # else:
        #     args.epoch_num = 401
        #     args.cache = True
        results = []
        results2 = []
        for repeat in range(args.repeat_num):
            result_val = []
            result_test = []

            result_test2 = []
            max_Epoch=[]
            Train_Time=[]
            maxTrainTime=[]
            Embedding=[]
            anchorset_ids=[]

            time1 = time.time()
            data_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)
            time2 = time.time()
            print(dataset_name, 'load time',  time2-time1)

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

            # data
            for i,data in enumerate(data_list):
                anchorset_ids=preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
                edge_labels=[]
                if args.AdverserialAttack:
                    addTestPairEdges(task, args, data, anchorset_ids)
                    # data_mask_link_positive_test2, data_mask_link_negative_test2,data_dists_max_advAttack, data_dists_argmaxadvAttack =createTestGraphForAdverseAttack(task,i,args,data,anchorset_ids,edge_labels, device='cpu')
                    #createTestGraphForAdverseAttack(task,i,args,data,anchorset_ids,edge_labels, device='cpu')
                    # print("Shape of original dist_max: ", data.dists_max.shape, "\t Number of non-zero values: ", np.sum([len(torch.nonzero(data.dists_max[i])) for i in range(len(data.dists_max[i]))]))
                    # print("Shape of adverse dist_max: ", data.dists_max_advAttack.shape, "\t Number of non-zero values: ", np.sum([len(torch.nonzero(data.dists_max_advAttack[i])) for i in range(len(data.dists_max_advAttack[i]))]))
                ###############################
                data = data.to(device)
                data_list[i] = data

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
                if epoch==200:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                model.train()
                optimizer.zero_grad()
                shuffle(data_list)
                effective_len = len(data_list)//args.batch_size*len(data_list)
                for id, data in enumerate(data_list[:effective_len]):
                    if args.permute:
                        anchorset_ids = preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                        if args.AdverserialAttack:
                            # since anchors change need to recompute the adversarial dist max
                            data.dists_max_advAttack, data.dists_argmaxadvAttack = get_dist_max(anchorset_ids, data.dists2, device)
                    out = model(data)
                    # get_link_mask(data,resplit=False)  # resample negative links
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
                            # if this is slow, no need to do this normalization
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad /= args.batch_size
                        optimizer.step()
                        optimizer.zero_grad()


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
                        # get_link_mask(data, resplit=False)  # resample negative links
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


                        ##########################################################################################################
                        # Test 1
                        if args.AdverserialAttack:
                            #print("Before modification: ", data.dists_max)
                            dists_tmp = data.dists.clone()
                            dists_max_temp = data.dists_max.clone()
                            dists_argmax_temp = data.dists_argmax.clone()
                            #print('Orignal :',data.mask_link_positive_test)
                            #print('Modified : ',data.mask_link_positive_test2)

                            # if(torch.all(torch.eq(data.dists, data.dists2))):
                            #     print("dist same")
                            # else:
                            #     print(data.dists.shape, data.dists2.shape)
                            # if(torch.all(torch.eq(data.dists_max, data.dists_max_advAttack))):
                            #         print("dist max same")
                            # else:
                            #     print(data.dists_max.shape, data.dists_max_advAttack.shape)
                            # if(torch.all(torch.eq(data.dists_argmax, data.dists_argmaxadvAttack))):
                            #     print("dists arg max same")
                            # else:
                            #     print(data.dists_argmax.shape, data.dists_argmaxadvAttack.shape)

                            # print("Difference: ", data.dists - data.dists2)
                            data.dists = data.dists2.clone()
                            data.dists_max = data.dists_max_advAttack.clone()
                            data.dists_argmax = data.dists_argmaxadvAttack.clone()
                            #print("After modification: ", data.dists_max)
                            #print("Difference: ", data.dists_max - dists_max_temp)

                            # if(torch.all(torch.eq(data.dists, data.dists2))):
                            #     print("dist same")
                            # else:
                            #     print(data.dists.shape, data.dists2.shape)
                            # if(torch.all(torch.eq(data.dists_max, data.dists_max_advAttack))):
                            #         print("dist max same")
                            # else:
                            #     print(data.dists_max.shape, data.dists_max_advAttack.shape)
                            # if(torch.all(torch.eq(data.dists_argmax, data.dists_argmaxadvAttack))):
                            #     print("dists arg max same")
                            # else:
                            #     print(data.dists_argmax.shape, data.dists_argmaxadvAttack.shape)

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

                            # edge_mask_test2 = np.concatenate((data_mask_link_positive_test2, data_mask_link_negative_test2), axis=-1)
                            # nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test2[0, :]).long().to(device))
                            # nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test2[1, :]).long().to(device))
                            # pred = torch.sum(nodes_first * nodes_second, dim=-1)
                            # label_positive = torch.ones([data_mask_link_positive_test2.shape[1], ], dtype=pred.dtype)
                            # label_negative = torch.zeros([data_mask_link_negative_test2.shape[1], ], dtype=pred.dtype)
                            # label = torch.cat((label_positive, label_negative)).to(device)
                            # loss_test2 += loss_func(pred, label).cpu().data.numpy()
                            # auc_test2 += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

                            data.dists = dists_tmp.clone()
                            data.dists_max = dists_max_temp.clone()
                            data.dists_argmax = dists_argmax_temp.clone()
                            # print("Reset: ", data.dists_max)
                            # exit()

                            #dists_max_temp=[]
                            # dists_argmax_temp=[]
                            #out = model(data)
                        ############################################################################################################

                        # Embedding.append(out)

                    loss_train /= id+1
                    loss_val /= id+1
                    loss_test /= id+1
                    emb_norm_min /= id+1
                    emb_norm_max /= id+1
                    emb_norm_mean /= id+1
                    auc_train /= id+1
                    auc_val /= id+1
                    auc_test /= id+1

                    if args.AdverserialAttack:
                        loss_test2 /= id+1
                        auc_test2 /= id+1

                    # if args.AdverserialAttack:
                    #     print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                    #       'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test) ,'ADVERSERIAL_ATTACK Test AUC: {:.4f}'.format(auc_test2))
                    # else:
                    #     print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                    #       'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test))


                    
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

                    if args.AdverserialAttack:
                        result_test2.append(auc_test2)


            result_val = np.array(result_val)
            result_test = np.array(result_test)
            results.append(result_test[np.argmax(result_val)])

            #FinalEmbedding=Embedding[np.argmax(result_val)]

            if args.AdverserialAttack:
                result_test2 = np.array(result_test2)
                results2.append(result_test2[np.argmax(result_val)])

        results = np.array(results)
        results_mean = np.mean(results).round(3)
        results_std = np.std(results).round(3)
        print('-----------------ROC AUC Scores-------------------')
        print(results_mean, results_std)
        print(results)

        if args.AdverserialAttack:
            results2 = np.array(results2)
            results2_mean = np.mean(results2).round(3)
            results2_std = np.std(results2).round(3)
            print('-----------------AdverserialAttack ROC AUC Scores-------------------')
            print(results2_mean, results2_std)
            print(results2)
        with open('results/{}_{}_{}_layer{}_approximate{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.approximate), 'w') as f:
            f.write('{}, {}\n'.format(results_mean, results_std))
            if args.AdverserialAttack:
                f.write('Adverserial Attack : ')
                f.write('{}, {}\n'.format(results_mean, results_std))
                f.write('Result : {}\n\n'.format(results2))
        # with open('results/EMBEDDING_{}_{}_{}_layer{}_approximate{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.approximate), 'w') as f:
        #     f.write('{}'.format(FinalEmbedding))

# export scalar data to JSON for external processing
writer_train.export_scalars_to_json("./all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json("./all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json("./all_scalars.json")
writer_test.close()