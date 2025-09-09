import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc, TransMIL
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_chief import CHIEF
from models.model_titan import TITAN
from models.model_ViLa_MIL import ViLa_MIL_Model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt', criteria=None):

        if not criteria:
            score = -val_loss
        else:
            score = criteria

        if self.best_score is None:
            self.best_score = -1
            self.save_checkpoint(val_loss, model, ckpt_name, criteria=criteria)
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, ckpt_name, criteria=criteria)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name, criteria=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if criteria:
                print(f'Validation criteria increased ({self.best_score:.6f} --> {criteria:.6f}).  Saving model ...')
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args, pseudo=False, notsavesplit=False, require_patient_results=True, disableAUC=False):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if not notsavesplit:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"conch_init": args.conch_init, "conch_freeze": args.conch_freeze})
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})

        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    elif args.model_type == 'transmil':
        model = TransMIL(**model_dict)
    elif args.model_type == 'abmil':
        model_dict.update({"conch_init": args.conch_init, "conch_freeze": args.conch_freeze})
        model = CLAM_SB(**model_dict, instance_loss_fn=None)
    elif args.model_type == 'vila':
        import ml_collections
        config = ml_collections.ConfigDict()
        if args.model_size == 'conch':
            config.input_size = 512
        else:
            config.input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        model_dict = {'config': config, 'num_classes': args.n_classes}
        model = ViLa_MIL_Model(**model_dict)
    elif args.model_type == 'chief':
        model = CHIEF(size_arg="small", dropout=True, 
                      n_classes=args.n_classes, anatomic=args.anatomic)
    elif args.model_type == 'titan':
        model = TITAN(num_classes=args.n_classes)
    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model_dict.update({"top_k": args.topk})
            model = MIL_fc(**model_dict)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    if not pseudo:
        train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample, batchsize=args.batch_size, vila=args.model_type == 'vila')
        val_loader = get_split_loader(val_split, testing=args.testing, vila=args.model_type == 'vila')
        test_loader = get_split_loader(test_split, testing=args.testing, vila=args.model_type == 'vila')
    else:
        train_loader = get_pseudo_loader_preload(train_split, training=True, testing=args.testing, batchsize=args.batch_size)
        val_loader = get_pseudo_loader_preload(val_split, testing=args.testing)
        test_loader = get_pseudo_loader_preload(test_split, testing=args.testing)
    
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=40, verbose=True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        print("\n\nCurrent Epoch {}".format(epoch))
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            print(f"lr: {scheduler.get_last_lr()}")
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, args.bag_size)
            scheduler.step()
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                                 early_stopping, writer, loss_fn, args.results_dir, disableAUC=disableAUC)
        
        elif args.model_type == 'vila':
            print(f"lr: {scheduler.get_last_lr()}")
            train_loop_vila(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.bag_size)
            scheduler.step()
            stop = validate_vila(cur, epoch, model, val_loader, args.n_classes,
                            early_stopping, writer, loss_fn, args.results_dir, disableAUC=disableAUC)

        else:
            print(f"lr: {scheduler.get_last_lr()}")
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.bag_size)
            scheduler.step()
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                            early_stopping, writer, loss_fn, args.results_dir, disableAUC=disableAUC)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if disableAUC:
        return {}, 0, 0, 0, 0
    
    if args.model_type == 'vila':
        summary_func = summary_vila
    else:
        summary_func = summary

    _, val_error, val_auc, _ = summary_func(model, val_loader, args.n_classes, require_patient_results=require_patient_results)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary_func(model, test_loader, args.n_classes, require_patient_results=require_patient_results)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    acc_list = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        acc_list.append(acc)

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    bacc = np.mean(acc_list)
    print("Test balanced accuracy: ", bacc)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None, bag_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        if Y_hat.size(0) > 1:
            acc_logger.log_batch(Y_hat.cpu(), label.cpu())
        else:
            acc_logger.log(Y_hat, label)
        # acc_logger.log(Y_hat, label)
        
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  instance_loss_value,
                                                                                                  total_loss.item()) )
                #   + 'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        # print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('train class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, bag_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        if type(data) == tuple:
            # data = (d.to(device) for d in data)
            data = tuple(d.to(device) for d in data)
            label = label.to(device)
        else:
            data, label = data.to(device), label.to(device)
        # if bag_size:
        #     patch_indices = torch.randperm(data.size(0))[:bag_size]
        #     data = data[patch_indices]

        logits, Y_prob, Y_hat, _, _ = model(data)

        if Y_hat.size(0) > 1:
            acc_logger.log_batch(Y_hat.cpu(), label.cpu())
        else:
            acc_logger.log(Y_hat, label)
        # acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            if type(data) == tuple:
                print('batch {}, loss: {:.4f}, batch_size:{}, bag_size: {}'.format(batch_idx, loss_value, data[0].size(0), data[0].size(1)))
            elif len(data.shape) > 2:
                print('batch {}, loss: {:.4f}, batch_size:{}, bag_size: {}'.format(batch_idx, loss_value, data.size(0), data.size(1)))
            else:
                print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop_vila(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, bag_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data_s, data_l, label) in enumerate(loader):
        data_s, data_l, label = data_s.to(device), data_l.to(device), label.to(device)
        Y_prob, Y_hat, loss = model(data_s, data_l, label)
        if Y_hat.size(0) > 1:
            acc_logger.log_batch(Y_hat.cpu(), label.cpu())
        else:
            acc_logger.log(Y_hat, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}'.format(batch_idx, loss_value))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None, disableAUC=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            # data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if type(data) == tuple:
                data = (d.to(device, non_blocking=True) for d in data)
            else:
                data = data.to(device, non_blocking=True)

            label = label.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if disableAUC:
        auc = 0
    else:
        if n_classes == 2:
            # print(labels)
            auc = roc_auc_score(labels, prob[:, 1])

        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    if disableAUC:
        bacc = 0
    else:
        acc_list = []
        print('\nValidation Set')
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            acc_list.append(acc)
        bacc = np.mean(acc_list)
        print("balanced accuracy: ", bacc)

    print('Val Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, bacc: {:.4f}'.format(val_loss, val_error, auc, bacc))

    if early_stopping:
        assert results_dir
        if disableAUC:
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)), criteria=auc)

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None, disableAUC=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']

            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if disableAUC:
        auc = 0
    else:
        if n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))

    if disableAUC:
        bacc = 0
    else:
        acc_list = []
        print('\nValidation Set')
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            acc_list.append(acc)

            if writer and acc is not None:
                writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
        bacc = np.mean(acc_list)
        print("balanced accuracy: ", bacc)

    print('Val Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, bacc: {:.4f}'.format(val_loss, val_error, auc, bacc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    if early_stopping:
        assert results_dir
        if disableAUC:
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)), criteria=auc)

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_vila(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None, disableAUC=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data_s, data_l, label) in enumerate(loader):
            data_s, data_l, label = data_s.to(device), data_l.to(device), label.to(device)
            Y_prob, Y_hat, loss = model(data_s, data_l, label)

            acc_logger.log(Y_hat, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if disableAUC:
        auc = 0
    else:
        if n_classes == 2:
            # print(labels)
            auc = roc_auc_score(labels, prob[:, 1])

        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    if disableAUC:
        bacc = 0
    else:
        acc_list = []
        print('\nValidation Set')
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            acc_list.append(acc)
        bacc = np.mean(acc_list)
        print("balanced accuracy: ", bacc)

    print('Val Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, bacc: {:.4f}'.format(val_loss, val_error, auc, bacc))

    if early_stopping:
        assert results_dir
        if disableAUC:
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)), criteria=auc)

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes, require_patient_results=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    if require_patient_results:
        slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        # data, label = data.to(device), label.to(device)
        if type(data) == tuple:
                # data = (d.to(device) for d in data)
                data = tuple(d.to(device) for d in data)
        else:
            data = data.to(device)
        label = label.to(device)
        if require_patient_results:
            slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        if require_patient_results:
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger


def summary_vila(model, loader, n_classes, require_patient_results=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    if require_patient_results:
        slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    # for batch_idx, (data, label) in enumerate(loader):
    for batch_idx, (data_s, data_l, label) in enumerate(loader):
        data_s, data_l, label = data_s.to(device), data_l.to(device), label.to(device)
        # data, label = data.to(device), label.to(device)
        if require_patient_results:
            slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            # logits, Y_prob, Y_hat, _, _ = model(data)
            Y_prob, Y_hat, loss = model(data_s, data_l, label)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        if require_patient_results:
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger