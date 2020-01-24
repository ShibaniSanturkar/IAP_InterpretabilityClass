import torch as ch
from torchvision.models import *
from robustness.tools import helpers
from robustness.datasets import DATASETS
from robustness.tools.label_maps import CLASS_DICT
from robustness import model_utils, datasets
from tqdm.autonotebook import tqdm

def load_model(arch, dataset=None):
    '''
    Load pretrained model with specified architecture.
    Args:
        arch (str): name of one of the pytorch pretrained models or 
                    "robust" for robust model
        dataset (dataset object): not None only for robust model
    Returns:
        model: loaded model
    '''
    
    if arch != 'robust':
        model = eval(arch)(pretrained=True).cuda()
        model.eval()
        pass
    else:
        model_kwargs = {
            'arch': 'resnet50',
            'dataset': dataset,
            'resume_path': f'./models/RestrictedImageNet.pt'
        }

        model, _ = model_utils.make_and_restore_model(**model_kwargs)
        model.eval()
        model = model.module.model
    return model

def load_dataset(dataset, batch_size, num_workers=1, data_path='./data'):
    '''
    Load pretrained model with specified architecture.
    Args:
        dataset (str): name of one of dataset 
                      ('restricted_imagenet' or 'imagenet')
        batch_size (int): batch size
        num_workers (int): number of workers
        data_path (str): path to data
    Returns:
        ds: dataset object
        loader: dataset loader
        norm: normalization function for dataset
        label_map: label map (class numbers to names) for dataset
    '''
    
    ds = DATASETS[dataset](data_path)
    _, loader = ds.make_loaders(num_workers, batch_size)
    normalization = helpers.InputNormalize(ds.mean, ds.std)
    label_map = CLASS_DICT['ImageNet'] if dataset == 'imagenet' else CLASS_DICT['RestrictedImageNet']
    label_map = {k: v.split(',')[0] for k, v in label_map.items()}
    return ds, loader, normalization, label_map

def forward_pass(mod, im, normalization=None):
    '''
    Compute model output (logits) for a batch of inputs.
    Args:
        mod: model
        im (tensor): batch of images
        normalization (function): normalization function to be applied on inputs
        
    Returns:
        op: logits of model for given inputs
    '''
    if normalization is not None:
        im_norm = normalization(im)
    else:
        im_norm = im
    op = mod(im_norm.cuda())
    return op

def get_gradient(mod, im, targ, normalization, custom_loss=None):
    '''
    Compute model gradients w.r.t. inputs.
    Args:
        mod: model
        im (tensor): batch of images
        normalization (function): normalization function to be applied on inputs
        custom_loss (function): custom loss function to employ (optional)
        
    Returns:
        grad: model gradients w.r.t. inputs
        loss: model loss evaluated at inputs
    '''    
    def compute_loss(inp, target, normalization):
        if custom_loss is None:
            output = forward_pass(mod, inp, normalization)
            return ch.nn.CrossEntropyLoss()(output, target.cuda())
        else:
            return custom_loss(mod, inp, target.cuda(), normalization)
        
    x = im.clone().detach().requires_grad_(True)
    loss = compute_loss(x, targ, normalization)
    grad, = ch.autograd.grad(loss, [x])
    return grad.clone(), loss.detach().item()

def visualize_gradient(t):
    '''
    Visualize gradients of model. To transform gradient to image range [0, 1], we 
    subtract the mean, divide by 3 standard deviations, and then clip.
    
    Args:
        t (tensor): input tensor (usually gradients)
    '''  
    mt = ch.mean(t, dim=[2, 3], keepdim=True).expand_as(t)
    st = ch.std(t, dim=[2, 3], keepdim=True).expand_as(t)
    return ch.clamp((t - mt) / (3 * st) + 0.5, 0, 1) 

def L2PGD(mod, im, targ, normalization, step_size, Nsteps,
        eps=None, targeted=True, custom_loss=None):
    '''
    Compute L2 adversarial examples for given model.
    Args:
        mod: model
        im (tensor): batch of images
        targ (tensor): batch of labels
        normalization (function): normalization function to be applied on inputs
        step_size (float): optimization step size
        Nsteps (int): number of optimization steps
        eps (float): radius of L2 ball
        targeted (bool): True if we want to maximize loss, else False
        custom_loss (function): custom loss function to employ (optional)
        
    Returns:
        x: batch of adversarial examples for input images
    '''      
    if custom_loss is None:
        loss_fn = ch.nn.CrossEntropyLoss()
    else:
        loss_fn = custom_loss 
        
    sign = -1 if targeted else 1
        
    it = tqdm(enumerate(range(Nsteps)), total=Nsteps)
    
    x = im.detach()
    l = len(x.shape) - 1
    
    for _, i in it:    
        x = x.clone().detach().requires_grad_(True)
        g, loss = get_gradient(mod, x, targ, normalization, 
                               custom_loss=custom_loss)
        
        it.set_description(f'Loss: {loss}')
        
        with ch.no_grad():
            
            # Compute gradient step 
            g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
            scaled_g = g / (g_norm + 1e-10)
            x += sign * scaled_g * step_size
            
            # Project back to L2 eps ball
            if eps is not None:
                diff = x - im
                diff = diff.renorm(p=2, dim=0, maxnorm=eps)
                x = im + diff
            x = ch.clamp(x, 0, 1)
    return x

def get_features(mod, im, normalization):
    '''
    Get feature representation of model  (output of layer before final linear 
    classifier) for given inputs.
    
    Args:
        mod: model
        im (tensor): batch of images
        targ (tensor): batch of labels
        normalization (function): normalization function to be applied on inputs
        
    Returns:
        features: batch of features for input images
    '''   
    feature_rep = ch.nn.Sequential(*list(mod.children())[:-1])
    im_norm = normalization(im.cpu()).cuda()
    features = feature_rep(im_norm)[:, :, 0, 0]
    return features
