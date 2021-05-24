from argparse import ArgumentParser
def make_args():
    parser = ArgumentParser()
    # general
    parser.add_argument('--comment', dest='comment', default='0', type=str,
                        help='comment')
    parser.add_argument('--task', dest='task', default='link', type=str,
                        help='link; link_pair')
    parser.add_argument('--model', dest='model', default='GraphReach', type=str,
                        help='model class name. E.g. GraphReach, GCN, SAGE, ...')
    parser.add_argument('--dataset', dest='dataset', default='All', type=str,
                        help='Set to All')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')
    parser.add_argument('--cache_no', dest='cache', action='store_false',
                        help='whether use cache')
    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help='whether use cpu')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)
    parser.add_argument('--attention', dest='attention', action='store_true',
                        help='whether use attention')

    # dataset
    parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.2, type=float)
    parser.add_argument('--rm_feature', dest='rm_feature', action='store_true',
                        help='whether rm_feature')
    parser.add_argument('--rm_feature_no', dest='rm_feature', action='store_false',
                        help='whether rm_feature')
    parser.add_argument('--permute', dest='permute', action='store_true',
                        help='whether permute subsets')
    parser.add_argument('--permute_no', dest='permute', action='store_false',
                        help='whether permute subsets')
    parser.add_argument('--feature_pre', dest='feature_pre', action='store_true',
                        help='whether pre transform feature')
    parser.add_argument('--feature_pre_no', dest='feature_pre', action='store_false',
                        help='whether pre transform feature')
    parser.add_argument('--dropout', dest='dropout', action='store_true',
                        help='whether dropout, default 0.5')
    parser.add_argument('--dropout_no', dest='dropout', action='store_false',
                        help='whether dropout, default 0.5')
    parser.add_argument('--select_anchors', dest='select_anchors', default='DiversifiedRandomK', type=str,
                        help='DiversifiedRandomK;DiversifiedTopK;topK; random')

    parser.add_argument('--batch_size', dest='batch_size', default=8, type=int) # implemented via accumulating gradient
    parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
    parser.add_argument('--feature_dim', dest='feature_dim', default=32, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=32, type=int)
    parser.add_argument('--output_dim', dest='output_dim', default=32, type=int)
    parser.add_argument('--anchor_num', dest='anchor_num', default=64, type=int)
    parser.add_argument('--normalize_adj', dest='normalize_adj', action='store_true',
                        help='whether normalize_adj')

    parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
    parser.add_argument('--epoch_num', dest='epoch_num', default=2001, type=int)
    parser.add_argument('--repeat_num', dest='repeat_num', default=2, type=int) # 10
    parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)


    #NODE2VEC ARGUMENTS
    parser.add_argument('--fastRandomWalk', dest='fastRandomWalk', action='store_true',
                        help='Default is NormalRandomWalk.')
    parser.set_defaults(fastRandomWalk=False)

    parser.add_argument('--attentionAddSelf', dest='attentionAddSelf', action='store_true',
                        help='Default is False.')
    parser.set_defaults(attentionAddSelf=False)
    
    parser.add_argument('--walk_length', type=int, default=20,
                    help='Length of walk per source. Default is 80.')

    parser.add_argument('--num_walks', type=int, default=50,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weightedRandomWalk', dest='weightedRandomWalk', action='store_true',
                        help='Default is unweightedRandomWalk.')
    parser.add_argument('--unweightedRandomWalk', dest='weightedRandomWalk', action='store_false')
    parser.set_defaults(weightedRandomWalk=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--normalized', dest='normalized', action='store_true',
                        help='Boolean specifying (un)normalized. Default is unnormalized.')
    parser.add_argument('--unnormalized', dest='normalized', action='store_false')
    parser.set_defaults(normalized=False)

    parser.add_argument('--edgelabel', dest='edgelabel', action='store_true',
                        help='whether use edgelabel')
    parser.add_argument('--noedgelabel', dest='edgelabel', action='store_false',
                        help='whether use edgelabel')

    parser.add_argument('--sampleXwalks', type=float, default=0.3,
                        help='Number of walks to be sampled for topK anchors')
    parser.add_argument('--sampleMbigraphs', type=int, default=5,
                        help='Number of bigraphs to be sampled')

    parser.add_argument('--deleteFedges', type=float, default=0.01,
                        help='Fraction of edges deleted to be deleted for Adversarial Attack')
    parser.add_argument('--addFedges', type=float, default=0.10,
                        help='Add false edges to nodes involved in sampled fraction of test pairs')

    parser.add_argument('--AdversarialAttack', dest='AdversarialAttack', action='store_true',
                        help='Boolean flag')

    parser.add_argument('--Num_Anchors', dest='Num_Anchors', default='logn2', type=str,
                        help='Num_Anchors')

    parser.set_defaults(gpu=True, task='link', model='GraphReach', dataset='All',select_anchors='DiversifiedRandomK',
                        cache=False, rm_feature=False,
                        permute=False, feature_pre=True, dropout=True,
                        normalize_adj=False,edgelabel=False,attention=False,AdversarialAttack=False)
    args = parser.parse_args()
    return args