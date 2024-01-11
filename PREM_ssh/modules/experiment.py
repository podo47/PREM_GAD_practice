import torch

from sklearn.metrics._ranking import roc_auc_score
from modules.model import Model
from modules.train import train_model, eval_model
from modules.utils import set_random_seeds

def run_experiment(args, seed, device, dataloader, ano_label, all_label):
    set_random_seeds(seed)
    # Create GGD model
    model = Model(
        g=dataloader.g,
        n_in=dataloader.en.shape[1],
        n_hidden=args.n_hidden,
        k=args.k
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_function = torch.nn.BCELoss()

    print(f"Seed {seed}")
    torch.cuda.reset_peak_memory_stats()
    state_path, stats, time_train = train_model(
        args, dataloader, model, optimizer, loss_function
    )
    mem_train = torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(state_path))
    torch.cuda.reset_peak_memory_stats()
    
    # Evaluation
    score, time_test = eval_model(args, dataloader, model)
    mem_test = torch.cuda.max_memory_allocated()

    # 個別看不同類別的 ano
    if args.type != 'all':
        for i in range(len(ano_label)):
            # 其中一個類別的ano為 0 但另一類的ano為 1 -> 整體ano labal為 1，須修正才能看單一類別ano下的效能
            if ano_label[i] == 0 and all_label[i] == 1:
                # 預設ano為 1的那一類的ano_score完全正確
                score[i] = 1.0

    auc = roc_auc_score(all_label, score)
        
    stats["mem_train"] = mem_train
    stats["mem_test"] = mem_test
    stats["time_train"] = time_train
    stats["time_test"] = time_test
    stats["AUC"] = auc

    return model, stats

