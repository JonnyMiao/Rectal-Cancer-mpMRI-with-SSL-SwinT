import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from monai.metrics import get_confusion_matrix, compute_confusion_matrix_metric
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from sklearn.metrics import auc as AUC
from sklearn.metrics import roc_curve


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


class Trainer:
    def __init__(self, args, train_loader, val_loader, model, loss_func, optimizer, scheduler, start_epoch, viz=None):
        self.viz = viz
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.loss_func = loss_func
        self.args = args
        self.env_name = args.dataset + '_' + args.therapy_type + '_' + args.task + '_' + args.roi + '_' + args.model_name
        self.label_array = pd.read_excel("../dataset/label.xlsx",
                                         sheet_name="Sheet1",
                                         dtype={'patient_id': str})

    def get_label(self, index):
        dict = self.label_array.set_index('patient_id').to_dict()[self.args.task]
        label = int(dict[index])

        return label

    def get_clinic(self, index):
        '''
        Get Clinic Information cT cN CRM Loc CEA
        :param index:
        :return:
        '''
        cT = self.label_array.set_index('patient_id').to_dict()['cT'][index]
        cN = self.label_array.set_index('patient_id').to_dict()['cN'][index]
        CRM = self.label_array.set_index('patient_id').to_dict()['CRM'][index]
        Loc = self.label_array.set_index('patient_id').to_dict()['FD'][index]
        CEA = self.label_array.set_index('patient_id').to_dict()['CEA-status'][index]
        clinic = torch.Tensor([cT, cN, CRM, Loc, CEA])
        clinic = torch.where(torch.isnan(clinic), torch.zeros_like(clinic), clinic)
        clinic = clinic.unsqueeze(dim=0)
        return clinic

    def prepare_input(self, batch_data):
        '''

        :param batch_data:
        :return:
        '''
        input1, input2, input3 = batch_data["images1"], batch_data["images2"], batch_data["images3"]
        input = torch.cat([input1, input2, input3], dim=1)
        clinic = torch.zeros([input.shape[0], 1, 5])
        label = torch.zeros([input.shape[0]])
        input = input.cuda()
        for i in range(input.shape[0]):
            index = os.path.basename(batch_data["images1_meta_dict"]["filename_or_obj"][i]).split('_')[0]
            label[i] = self.get_label(index)
            clinic[i,] = self.get_clinic(index)
        label = label.long().cuda()
        clinic = clinic.cuda()
        return input, clinic, label

    def run_training(self):
        scaler = None
        best_model = None
        if self.args.amp:
            scaler = GradScaler()
        if self.args.out_channels == 3:
            val_acc_max = np.zeros([3, 7])
        else:
            val_acc_max = torch.zeros([8])
        for epoch in range(self.start_epoch, self.args.max_epochs + 1):
            print(time.ctime(), 'Epoch:', epoch)
            epoch_time = time.time()
            train_loss = self.train_epoch(epoch, scaler)
            self.viz.line([train_loss],
                          [epoch],
                          env=self.env_name,
                          win="Train_Loss",
                          opts=dict(title="Train_Loss", xlabel='Epoch'),
                          update='append')

            if (epoch + 1) % self.args.val_every == 0:
                epoch_time = time.time()
                val_loss, val_acc = self.val_epoch(epoch)
                self.viz.line([val_loss],
                              [epoch],
                              env=self.env_name,
                              win="Val_Loss",
                              opts=dict(title="Val_Loss", xlabel='Epoch'),
                              update='append')
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
            self.viz.line([lr],
                          [epoch],
                          env=self.env_name,
                          win="Learning_Rate",
                          opts=dict(title="Learning_Rate", xlabel='Epoch'),
                          update='append')
            if self.args.out_channels == 3:
                print('Final validation  {}/{}'.format(epoch, self.args.max_epochs - 1),
                      'auc', np.mean(val_acc[1, :]), 'time {:.2f}s'.format(time.time() - epoch_time))
                if np.mean(val_acc[1, :]) >= np.mean(val_acc_max[1, :]):
                    print(
                        'new best auc ({:.6f} --> {:.6f}). '.format(np.mean(val_acc[1, :]),
                                                                    np.mean(val_acc_max[1, :])))
                    best_model = copy.deepcopy(self.model)
                    val_acc_max = val_acc
                    self.save_checkpoint(self.args.save, self.model, epoch, self.args,
                                         best_acc=val_acc_max,
                                         optimizer=self.optimizer,
                                         scheduler=self.scheduler,
                                         filename='model_best.pt')

                self.save_checkpoint(self.args.save, self.model, epoch, self.args,
                                     best_acc=val_acc,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     filename='model_final.pt')
            else:
                print('Final validation  {}/{}'.format(epoch, self.args.max_epochs - 1),
                      'auc', val_acc[-1], 'time {:.2f}s'.format(time.time() - epoch_time))
                if val_acc[-1] >= val_acc_max[-1]:
                    print('new best auc ({:.6f} --> {:.6f}). '.format(val_acc_max[-1], val_acc[-1]))
                    best_model = copy.deepcopy(self.model)
                    val_acc_max = val_acc
                    self.save_checkpoint(self.args.save, self.model, epoch, self.args,
                                         best_acc=val_acc_max,
                                         optimizer=self.optimizer,
                                         scheduler=self.scheduler,
                                         filename='model_best.pt')

                self.save_checkpoint(self.args.save, self.model, epoch, self.args,
                                     best_acc=val_acc,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     filename='model_final.pt')

            print('Final training  {}/{}'.format(epoch, self.args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'Lr: {}'.format(lr[0]),
                  'time {:.2f}s'.format(time.time() - epoch_time),
                  'left_time {:.2f}h'.format((time.time() - epoch_time) * (self.args.max_epochs - epoch) / 3600))

        return best_model, val_acc_max

    def compute_metrics(self, output, target):
        '''

        :param output:
        :param target:
        :return:
        '''
        if self.args.out_channels == 2:
            boutput = (output == torch.max(output, dim=1, keepdim=True).values).float()
            confusion_matrix = torch.sum(get_confusion_matrix(boutput, target, include_background=False), dim=0)
            accuracy = float(compute_confusion_matrix_metric(metric_name="accuracy", confusion_matrix=confusion_matrix))
            balanced_accuracy = float(compute_confusion_matrix_metric(metric_name="balanced accuracy",
                                                                      confusion_matrix=confusion_matrix))
            sensitivity = float(
                compute_confusion_matrix_metric(metric_name="sensitivity", confusion_matrix=confusion_matrix))
            specificity = float(
                compute_confusion_matrix_metric(metric_name="specificity", confusion_matrix=confusion_matrix))
            ppv = float(compute_confusion_matrix_metric(metric_name="precision", confusion_matrix=confusion_matrix))
            npv = float(compute_confusion_matrix_metric(metric_name="negative predictive value",
                                                        confusion_matrix=confusion_matrix))
            f1_score = float(compute_confusion_matrix_metric(metric_name="f1 score", confusion_matrix=confusion_matrix))

            fpr, tpr, thresholds = roc_curve(target[:, 1], output[:, 1])
            auc = AUC(fpr, tpr)
            # draw roc curve
            # plt.plot(fpr, tpr,
            #          label=f'{self.args.task} (AUC = {auc:0.3f})',
            #          )
            # plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            #
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title(f'ROC Curves for {self.args.task}')
            # plt.legend(loc='lower right')
            # plt.grid(True)
            # plt.show()

            Show_Metric = torch.Tensor(
                [accuracy, balanced_accuracy, sensitivity, specificity, ppv, npv, f1_score, auc]).detach().cpu().numpy()
            row = {"Accuracy": accuracy,
                   "Balanced_Acc": balanced_accuracy,
                   "Sensitivity": sensitivity,
                   "Specificity": specificity,
                   "PPV": ppv,
                   "NPV": npv,
                   "F1_Score": f1_score,
                   "AUC": auc}
        else:
            boutput = (output == torch.max(output, dim=1, keepdim=True).values).float()
            confusion_matrix = torch.sum(get_confusion_matrix(boutput, target, include_background=True), dim=0)
            accuracy = compute_confusion_matrix_metric(metric_name="accuracy",
                                                       confusion_matrix=confusion_matrix).unsqueeze(
                dim=0)
            balanced_accuracy = compute_confusion_matrix_metric(metric_name="balanced accuracy",
                                                                confusion_matrix=confusion_matrix).unsqueeze(dim=0)
            sensitivity = compute_confusion_matrix_metric(metric_name="sensitivity",
                                                          confusion_matrix=confusion_matrix).unsqueeze(dim=0)
            specificity = compute_confusion_matrix_metric(metric_name="specificity",
                                                          confusion_matrix=confusion_matrix).unsqueeze(dim=0)
            ppv = compute_confusion_matrix_metric(metric_name="precision",
                                                  confusion_matrix=confusion_matrix).unsqueeze(dim=0)
            npv = compute_confusion_matrix_metric(metric_name="negative predictive value",
                                                  confusion_matrix=confusion_matrix).unsqueeze(dim=0)
            f1_score = compute_confusion_matrix_metric(metric_name="f1 score",
                                                       confusion_matrix=confusion_matrix).unsqueeze(dim=0)

            y_true, y_pred = target.numpy(), output.numpy()
            n_classes = y_true.shape[1]

            fpr = dict()
            tpr = dict()
            roc_auc = []
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
                roc_auc.append(AUC(fpr[i], tpr[i]))
            # classed_name = ['TRG 0', 'TRG 1-2', 'TRG 3']
            # plt.plot(
            #     fpr[i], tpr[i],
            #     label=f'{classed_name[i]} (AUC = {AUC(fpr[i], tpr[i]):0.3f})',
            #     # color=label_info['color'],
            #     linestyle='--' if i == 0 else '-'
            # )
            # plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            #
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title(f'ROC Curves for {self.args.task}')
            # plt.legend(loc='lower right')
            # plt.grid(True)
            # plt.show()
            row = {"Accuracy": float(torch.mean(accuracy)),
                   "Balanced_Acc": float(torch.mean(balanced_accuracy)),
                   "Sensitivity": float(torch.mean(sensitivity)),
                   "Specificity": float(torch.mean(specificity)),
                   "PPV": float(torch.mean(ppv)),
                   "NPV": float(torch.mean(npv)),
                   "F1_Score": float(torch.mean(f1_score)),
                   "AUC": np.array(roc_auc).mean()
                   }
            Show_Metric = torch.cat(
                [accuracy, balanced_accuracy, sensitivity, specificity, ppv, npv, f1_score,
                 torch.tensor(roc_auc).unsqueeze(dim=0)],
                dim=0).detach().cpu().numpy()
        return Show_Metric, row

    def train_epoch(self, epoch, scaler):
        self.model.train()
        start_time = time.time()
        run_loss = AverageMeter()
        output_all = torch.zeros([len(self.train_loader.dataset), self.args.out_channels])
        target_all = torch.zeros([len(self.train_loader.dataset), self.args.out_channels])

        for idx, batch_data in enumerate(self.train_loader):
            for param in self.model.parameters():
                param.grad = None

            data, clinic, target = self.prepare_input(batch_data)
            logits = self.model(data, clinic)

            loss = self.loss_func(logits, target)

            output = F.softmax(logits, dim=1).detach()
            target = F.one_hot(target.long(), num_classes=self.args.out_channels)
            assert output.shape == target.shape

            start = idx * self.args.batch_size
            end = (idx + 1) * self.args.batch_size
            end = None if end > len(self.train_loader.dataset) else end

            output_all[start:end, ] = output
            target_all[start:end, ] = target

            loss.backward()
            self.optimizer.step()

            run_loss.update(loss.item(), n=self.args.batch_size)

            print('Epoch {}/{} {}/{}'.format(epoch, self.args.max_epochs, idx, len(self.train_loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time)
                  )

            start_time = time.time()

        Show_Train_Metric, row = self.compute_metrics(output_all, target_all)
        if self.args.out_channels == 2:
            self.viz.line([Show_Train_Metric], [epoch], env=self.env_name, win="Train_Metric",
                          opts=dict(title="Train_Metric", xlabel='Epoch',
                                    legend=["Accuracy", "Balanced_Acc", "Sensitivity",
                                            "Specificity", "PPV", "NPV", "F1_Score", "AUC"]),
                          update='append')
        elif self.args.out_channels == 3:
            le = ["TRG-0", "TRG-12", "TRG-3"] if self.args.task == 'label_TRG' else ["NAR-0", "NAR-1", "NAR-2"]
            self.viz.line([Show_Train_Metric[0]], [epoch], env=self.env_name, win="Train_Metric1",
                          opts=dict(title="Train_Accuracy", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Train_Metric[1]], [epoch], env=self.env_name, win="Train_Metric2",
                          opts=dict(title="Train_Balanced_Acc", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Train_Metric[2]], [epoch], env=self.env_name, win="Train_Metric3",
                          opts=dict(title="Train_Sensitivity", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Train_Metric[3]], [epoch], env=self.env_name, win="Train_Metric4",
                          opts=dict(title="Train_Specificity", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Train_Metric[4]], [epoch], env=self.env_name, win="Train_Metric5",
                          opts=dict(title="Train_PPV", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Train_Metric[5]], [epoch], env=self.env_name, win="Train_Metric6",
                          opts=dict(title="Train_NPV", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Train_Metric[6]], [epoch], env=self.env_name, win="Train_Metric7",
                          opts=dict(title="Train_F1_Score", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Train_Metric[7]], [epoch], env=self.env_name, win="Train_Metric8",
                          opts=dict(title="Train_AUC", xlabel='Epoch',
                                    legend=le),
                          update='append')
        pd.set_option("display.max_columns", 1000)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_columns", 1000)
        overall_metric = pd.DataFrame.from_dict(row, orient="index", columns=["Train_Metric"]).T

        print('Train_Epoch {}/{}\n'.format(epoch, self.args.max_epochs),
              overall_metric)

        for param in self.model.parameters():
            param.grad = None
        return run_loss.avg

    def val_epoch(self, epoch):
        self.model.eval()
        val_loss = AverageMeter()
        output_all = torch.zeros([len(self.val_loader.dataset), self.args.out_channels])
        target_all = torch.zeros([len(self.val_loader.dataset), self.args.out_channels])
        with torch.no_grad():
            for idx, batch_data in enumerate(self.val_loader):

                data, clinic, target = self.prepare_input(batch_data)
                logits = self.model(data, clinic)

                loss = self.loss_func(logits, target)
                output = F.softmax(logits, dim=1).detach()
                target = F.one_hot(target.long(), num_classes=self.args.out_channels)
                assert output.shape == target.shape

                start = idx * self.args.batch_size
                end = (idx + 1) * self.args.batch_size
                end = None if end > len(self.val_loader.dataset) else end

                output_all[start:end, ] = output
                target_all[start:end, ] = target

                val_loss.update(loss.item(), n=data.shape[0])

        Show_Val_Metric, row = self.compute_metrics(output_all, target_all)
        if self.args.out_channels == 2:
            self.viz.line([Show_Val_Metric], [epoch], env=self.env_name, win="Val_Metric",
                          opts=dict(title="Val_Metric", xlabel='Epoch',
                                    legend=["Accuracy", "Balanced_Acc", "Sensitivity",
                                            "Specificity", "PPV", "NPV", "F1_Score", "AUC"]),
                          update='append')
        elif self.args.out_channels == 3:
            le = ["TRG-0", "TRG-12", "TRG-3"] if self.args.task == 'label_TRG' else ["NAR-0", "NAR-1", "NAR-2"]
            self.viz.line([Show_Val_Metric[0]], [epoch], env=self.env_name, win="Val_Metric1",
                          opts=dict(title="Val_Accuracy", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Val_Metric[1]], [epoch], env=self.env_name, win="Val_Metric2",
                          opts=dict(title="Val_Balanced_Acc", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Val_Metric[2]], [epoch], env=self.env_name, win="Val_Metric3",
                          opts=dict(title="Val_Sensitivity", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Val_Metric[3]], [epoch], env=self.env_name, win="Val_Metric4",
                          opts=dict(title="Val_Specificity", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Val_Metric[4]], [epoch], env=self.env_name, win="Val_Metric5",
                          opts=dict(title="Val_PPV", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Val_Metric[5]], [epoch], env=self.env_name, win="Val_Metric6",
                          opts=dict(title="Val_NPV", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Val_Metric[6]], [epoch], env=self.env_name, win="Val_Metric7",
                          opts=dict(title="Val_F1_Score", xlabel='Epoch',
                                    legend=le),
                          update='append')
            self.viz.line([Show_Val_Metric[7]], [epoch], env=self.env_name, win="Val_Metric8",
                          opts=dict(title="Val_AUC", xlabel='Epoch',
                                    legend=le),
                          update='append')
        pd.set_option("display.max_columns", 1000)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_columns", 1000)
        overall_metric = pd.DataFrame.from_dict(row, orient="index", columns=["Val_Metric"]).T
        print('Val_Epoch {}/{}\n'.format(epoch, self.args.max_epochs), overall_metric)

        return val_loss.avg, Show_Val_Metric

    def save_checkpoint(self, directory, model, epoch, args, best_acc, filename='model.pt', optimizer=None,
                        scheduler=None):
        if not os.path.exists(directory):
            os.makedirs(directory)
        state_dict = model.state_dict()
        save_dict = {
            'epoch': epoch,
            'Acc': best_acc,
            'model_state_dict': state_dict
        }
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            save_dict['lr_scheduler_state_dict'] = scheduler.state_dict()
        savename = os.path.join(directory, filename)
        torch.save(save_dict, savename)
        print('Saving checkpoint', savename)
