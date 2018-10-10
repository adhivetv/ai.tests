import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, dim, ignore_thres, img_dim):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    dim = dim
    mask        = torch.zeros(nB, nA, dim, dim)
    conf_mask   = torch.ones(nB, nA, dim, dim)
    tx          = torch.zeros(nB, nA, dim, dim)
    ty          = torch.zeros(nB, nA, dim, dim)
    tw          = torch.zeros(nB, nA, dim, dim)
    th          = torch.zeros(nB, nA, dim, dim)
    tconf       = torch.zeros(nB, nA, dim, dim)
    tcls        = torch.zeros(nB, nA, dim, dim, num_classes)
    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * dim
            gy = target[b, t, 2] * dim
            gw = target[b, t, 3] * dim
            gh = target[b, t, 4] * dim
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            tconf[b, best_n, gj, gi] = 1
            if iou > 0.5:
                nCorrect += 1
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def create_modules(module_defs):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))
        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample( scale_factor=int(module_def['stride']),
                                    mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())
        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module('yolo_%d' % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
    return module_list
	

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    def forward(self, x, targets=None):
        bs = x.size(0)
        g_dim = x.size(2)
        stride =  self.img_dim / g_dim
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        prediction = x.view(bs,  self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0, 1, 3, 4, 2).contiguous()
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        # Calculate offsets for each grid
        grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape)
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        # Training
        if targets is not None:
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes.cpu().data,
                                                                            targets.cpu().data,
                                                                            scaled_anchors,
                                                                            self.num_anchors,
                                                                            self.num_classes,
                                                                            g_dim,
                                                                            self.ignore_thres,
                                                                            self.img_dim)
            nProposals = int((conf > 0.25).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            # Handle masks
            mask = Variable(mask.type(FloatTensor))
            cls_mask = Variable(mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_classes).type(FloatTensor))
            conf_mask = Variable(conf_mask.type(FloatTensor))
            # Handle target variables
            tx    = Variable(tx.type(FloatTensor), requires_grad=False)
            ty    = Variable(ty.type(FloatTensor), requires_grad=False)
            tw    = Variable(tw.type(FloatTensor), requires_grad=False)
            th    = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls  = Variable(tcls.type(FloatTensor), requires_grad=False)
            # Mask outputs to ignore non-existing objects
            loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2
            loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall
        else:
            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data


class YOLO(nn.Module):
    """YOLOv3 object detection model"""
    def __init__(self, cfg):
        super(YOLO, self).__init__()
        self.module_list = create_modules(cfg)
        self.seen = 0
        self.module_defs = cfg
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']
    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)
        self.losses['recall'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)
    def save_weights(self, path, cutoff=-1):
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)
    fp.close()
	def save_model(self, path, cutoff=-1):
        self.header_info[3] = self.seen
        torch.save((self.header_info, self.module_list), path)