import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt

def generate_input_generation_examples(loader, debug = True):
    loader.dataset.debug = debug
    image_batch, target_batch, \
    permutation_postion_embedding, \
    target_name_batch, permutation_label = next(iter(loader))
    target_name_batch = np.array(list(target_name_batch))
    return image_batch , target_batch, permutation_postion_embedding, target_name_batch, permutation_label


def add_model_weights_as_histogram(model, tb_writer, epoch):
    """
    Get named parameters and plot as histogram
    """
    for name, param in model.named_parameters():
        tb_writer.add_histogram(name.replace('.', '/'), param.data.cpu().abs(), epoch)
    return

def add_network_graph_tensorboard(model, inputs, tb_writer):
    tb_writer.add_graph(model, inputs)
    return

def get_random_inputs_labels(data_set, n=100):
    """
    get random inputs and labels
    """
    images = torch.Tensor([])
    target_labels = torch.Tensor([])
    target_labels_names = torch.Tensor([])

    while images.shape[0] < n:
        image_batch, target_batch, \
        permutation_postion_embedding, \
        target_name_batch, permutation_label = \
            generate_input_generation_examples(data_set, debug = False)

        if images.shape[0] + image_batch.shape[0] < n:
            pass
        else:
            amount_of_images_to_complete = n - images.shape[0]
            rand_indices = torch.randperm(amount_of_images_to_complete)
            import random
            rand_indices = random.sample(range(image_batch.shape[0]), amount_of_images_to_complete)
            image_batch = image_batch[rand_indices,::]
            target_batch = target_batch[rand_indices,::]
            target_name_batch = target_name_batch[rand_indices]
        if images.shape[0]:
            images = torch.concat([images, image_batch])
            target_labels = torch.concat([target_labels, target_batch])
            target_labels_names = np.append(target_labels_names, target_name_batch)
        else:
            images = image_batch
            target_labels = target_batch
            target_labels_names = target_name_batch
    return images, target_labels, target_labels_names

def add_data_embedings(dataset, tb_writer, n=100, global_step=1, tag="embedings"):
    """
    Add a few inputs and labels to tensorboard. 
    """
    
    images, target_labels, target_labels_names = get_random_inputs_labels(dataset, n=n)
    
    # Add image as embedding to tensorboard
    flatten_size = np.prod(list(images.shape[1::]))
    """
    mat - should be tensor
    label_img - should be tensor resize to smaller image size
    metadata - list of label data
    """
    tb_writer.add_embedding(mat = images.view(-1, flatten_size), 
                            metadata=list(target_labels_names), 
                            label_img= images,
                            global_step=global_step,
                            tag=tag)
    return

def get_target_and_prob(model, dataloader, device):
    """
    get targets and prediction probabilities
    """
    
    pred_prob = []
    targets = []
    for _, (data, target, permutation_postion_embedding, target_name, permutation_label) in enumerate(dataloader):
        
        _, prob = prediction(model, device, data, max_prob=False)
        
        pred_prob.append(prob)
        
        target = target.numpy()
        targets.append(target)
        
    targets = np.concatenate(targets)
    targets = targets.astype(int)
    pred_prob = np.concatenate(pred_prob, axis=0)
    
    return targets, pred_prob
    
def add_pr_curves_to_tensorboard(model, dataloader, device, tb_writer, epoch, num_classes=13):
    """
    Add precession and recall curve to tensorboard.
    """
    class_list = dataloader.dataset.class_list

    targets, pred_prob = get_target_and_prob(model, dataloader, device)
    
    for cls_idx in range(num_classes):
        binary_target = targets == cls_idx
        true_prediction_prob = pred_prob[:, cls_idx]
        
        tb_writer.add_pr_curve(class_list[cls_idx], 
                               targets[:, cls_idx], 
                               true_prediction_prob, 
                               global_step=epoch)
        
    return
def prediction(model, device, batch_input, max_prob=True):
    """
    get prediction for batch inputs
    """
    
    # send model to cpu/cuda according to your system configuration
    model.to(device)
    
    # it is important to do model.eval() before prediction
    model.eval()

    data = batch_input.to(device)
    with torch.no_grad():
        output = model(data)

    # get probability score using softmax
    prob = F.softmax(output, dim=1)
    
    if max_prob:
        # get the max probability
        pred_prob = prob.data.max(dim=1)[0]
    else:
        pred_prob = prob.data
    
    # get the index of the max probability
    pred_index = prob.data.max(dim=1)[1]
    
    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()


def add_wrong_prediction_to_tensorboard(model, dataloader, device, tb_writer, 
                                        epoch, tag='Wrong_Predections', max_images='all'):
    """
    Add wrong predicted images to tensorboard.
    """
    
    class_list = dataloader.dataset.class_list
    #number of images in one row
    num_images_per_row = 8
    im_scale = 3
    
    plot_images = []
    wrong_labels = []
    pred_prob = []
    right_label = []
    
    for _, (data, target, permutation_postion_embedding, target_name, permutation_label) in enumerate(dataloader):
        
        target = target.argmax(1)
        images = data.numpy()
        pred, prob = prediction(model, device, data)
        target = target.numpy()
        indices = pred.astype(int) != target.astype(int)
        
        plot_images.append(images[indices])
        wrong_labels.append(pred[indices])
        pred_prob.append(prob[indices])
        right_label.append(target[indices])
        if not isinstance(max_images, str) and len(right_label) > max_images:
            break
        
    plot_images = np.concatenate(plot_images, axis=0).squeeze()
    wrong_labels = np.concatenate(wrong_labels)
    wrong_labels = wrong_labels.astype(int)
    right_label = np.concatenate(right_label)
    right_label = right_label.astype(int)
    pred_prob = np.concatenate(pred_prob)
    
    
    if max_images == 'all':
        num_images = len(images)
    else:
        num_images = min(len(plot_images), max_images)
        
    fig_width = num_images_per_row * im_scale
    
    if num_images % num_images_per_row == 0:
        num_row = num_images//num_images_per_row
    else:
        num_row = int(num_images/num_images_per_row) + 1
        
    fig_height = num_row * im_scale
        
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)
    fig = plt.figure()
    plot_images =  plot_images.transpose(0,2,3,1)
    for i in range(num_images):
        num_row = (i//num_images_per_row) + 1
        
        plt.subplot(num_row, num_images_per_row, i+1, xticks=[], yticks=[])
        curr_image = plot_images[i]
        curr_image = np.uint8(curr_image)
        plt.imshow(curr_image)
        plt.gca().set_title('{0}({1:.2}), {2}'.format(class_list[wrong_labels[i]], 
                                                          pred_prob[i], 
                                                          class_list[right_label[i]]))
        
    tb_writer.add_figure(tag, fig, global_step=epoch)
    
    return