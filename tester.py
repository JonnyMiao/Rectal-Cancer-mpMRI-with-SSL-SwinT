import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import SimpleITK as sitk
from sklearn.metrics import auc as AUC
from sklearn.metrics import roc_curve
from monai.metrics import get_confusion_matrix, compute_confusion_matrix_metric

import matplotlib.pyplot as plt


class Tester:
    """
    Tester class
    """

    def __init__(self, viz, args, model, test_data_loader=None):
        self.viz = viz
        self.args = args
        self.model = model
        self.test_loader = test_data_loader
        self.trg_dict = {
            0: "TRG 0",
            1: "TRG 1-2",
            2: ">= TRG 3",
        }
        self.env_name = args.dataset + '_' + args.therapy_type + '_' + args.task + '_' + args.roi + '_' + args.model_name
        self.label_array = pd.read_excel("../dataset/label.xlsx",
                                         sheet_name="Sheet1", dtype={'patient_id': str}, )

    def save_as_nii(self, array, origin, direction, spacing, save_name=None):
        Img = sitk.GetImageFromArray(array)
        Img.SetOrigin(origin)
        Img.SetDirection(direction)
        Img.SetSpacing(spacing)
        sitk.WriteImage(Img, save_name)

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

    def save_to_file(self, file_name, contents):
        fh = open(file_name, 'a', encoding='utf-8')
        fh.write(contents)
        fh.close()

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
            # 计算FPR和TPR
            fpr, tpr, thresholds = roc_curve(target[:, 1], output[:, 1])
            # 计算AUC
            auc = AUC(fpr, tpr)
            plt.plot(fpr, tpr,
                     label=f'{self.args.task} (AUC = {auc:0.3f})',
                     )
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for {self.args.task}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

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
            classed_name = ['TRG 0', 'TRG 1-2', 'TRG 3']
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
                roc_auc.append(AUC(fpr[i], tpr[i]))
                plt.plot(
                    fpr[i], tpr[i],
                    label=f'{classed_name[i]} (AUC = {AUC(fpr[i], tpr[i]):0.3f})',
                    # color=label_info['color'],
                    linestyle='--' if i == 0 else '-'
                )
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for {self.args.task}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()
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
                [accuracy, balanced_accuracy, sensitivity, specificity, ppv, npv, f1_score, torch.tensor(roc_auc).unsqueeze(dim=0)],
                dim=0).detach().cpu().numpy()
        return Show_Metric, row

    def running_test(self):
        self.model.eval()
        predictions = {}
        output_all = torch.zeros([len(self.test_loader.dataset), self.args.out_channels])
        target_all = torch.zeros([len(self.test_loader.dataset), self.args.out_channels])
        with torch.no_grad():
            for idx, batch_data in enumerate(self.test_loader):

                data, clinic, target = self.prepare_input(batch_data)
                logits = self.model(data, clinic)

                output = F.softmax(logits, dim=1).detach()
                target = F.one_hot(target.long(), num_classes=self.args.out_channels)
                assert output.shape == target.shape
                output_all[idx,] = output
                target_all[idx,] = target

                index = os.path.basename(batch_data['images1'].meta["filename_or_obj"][0]).split('_')[0]
                if self.args.out_channels == 2:
                    predictions[index] = float(output[0, 1])
                else:
                    predictions[index] = output[0]

            Show_Test_Metric, row = self.compute_metrics(output_all, target_all)
            if self.args.out_channels == 3:
                indices = ['Accuracy', 'Balanced Accuracy', 'Sensitivity',
                           'Specificity', 'PPV', 'NPV', 'F1_Score', 'AUC']
                if self.args.task == 'label_TRG':
                    targets = ["TRG-0", "TRG-12", "TRG-3"]
                overall_metric = pd.DataFrame(Show_Test_Metric, index=indices, columns=targets)
                overall_metric['mean'] = overall_metric.mean(axis=1)
                overall_metric['std'] = overall_metric.std(axis=1)
                overall_metric = overall_metric.round(4)

                self.viz.bar(X=Show_Test_Metric, env=self.env_name, win="Test_Metric_Bar",
                             opts=dict(stacked=False, title='Test_Metric_Bar',
                                       xlabel=indices,
                                       legend=targets))
            else:
                overall_metric = pd.DataFrame.from_dict(row, orient="index", columns=["Test_Metric"]).T

            pd.set_option("display.max_columns", 1000)
            pd.set_option("display.width", 1000)
            pd.set_option("display.max_columns", 1000)
            print('Test\n', overall_metric)
            if not os.path.exists(self.args.save):
                self.args.save = self.args.checkpoint.split('model_')[0]
            formatters = ["{:^6}".format] * len(overall_metric.columns)
            with open(os.path.join(self.args.save, 'result.txt'), 'a') as f:
                f.write(overall_metric.round(4).to_string())
