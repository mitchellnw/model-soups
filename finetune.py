import argparse
import os
import torch
import clip
import os
from tqdm import tqdm
import time

from timm.data.transforms_factory import transforms_imagenet_train

from datasets.imagenet import ImageNet98p, ImageNet
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr
from zeroshot import zeroshot_classifier
from openai_imagenet_template import openai_imagenet_template
from sparse_linear_layer import SparseLinearLayer


# this goes through and replaces all linear layers with linear_replacement
# it also copies the weights from the old linear layer to the new one
# and sets the scores to be weights.abs() which seems reasonable to me for fine-tuning
# also set requires_grad false for the weights and bias
def replace_linear(model, linear_replacement, skip_modules=["lm_head", "conv1", "embedding"]):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            # For now we just replace the c_fc and c_proj layers.
            if name in ['c_fc', 'c_proj']:
                old_module = model._modules[name]
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                )

                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias)
                model._modules[name].scores.data.copy_(old_module.weight.data.abs())

                model._modules[name].weight.requires_grad = False
                model._modules[name].bias.requires_grad = False

    return model



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('/data/yfcc-tmp/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('checkpoints_folder'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prune-rate",
        type=float,
        default=0.5,
        help="0 means no pruning, 1 means all the pruning"
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    parser.add_argument(
        "--name",
        default='finetune_cp',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--timm-aug", action="store_true", default=False,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = 'cuda'

    if args.custom_template:
        template = [lambda x : f"a photo of a {x}."]
    else:
        template = openai_imagenet_template

    base_model, preprocess = clip.load(args.model, 'cuda', jit=False)
    # 98p is the 98% of ImageNet train set that we train on -- the other 2% is hodl-out val.
    if args.timm_aug:
        train_preprocess = transforms_imagenet_train(
                img_size=base_model.visual.input_resolution,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
    else:
        train_preprocess = preprocess
    train_dset = ImageNet98p(train_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    test_dset = ImageNet(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    clf = zeroshot_classifier(base_model, train_dset.classnames, template, DEVICE)
    NUM_CLASSES = len(train_dset.classnames)
    feature_dim = base_model.visual.output_dim

    # go through the model and swap out the linear layers with sparse linear layers
    replace_linear(base_model.visual, SparseLinearLayer)
    base_model.apply(lambda m : setattr(m, 'prune_rate', args.prune_rate))

    model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    # turn off the grad to everything that is not scores
    for n, p in model.named_parameters():
        if 'scores' not in n:
            p.requires_grad = False

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    print('optimizing {} params'.format(len(model_parameters)))
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    num_batches = len(train_dset.train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    loss_fn = torch.nn.CrossEntropyLoss()

    # model_path = os.path.join(args.model_location, f'{args.name}_0.pt')
    # print('Saving model to', model_path)
    # torch.save(model.module.state_dict(), model_path)

    for epoch in range(args.epochs):
        # Train
        model.train()
        end = time.time()
        for i, batch in enumerate(train_dset.train_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)
            data_time = time.time() - end

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(train_dset.train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_dset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # #Evaluate
        test_loader = test_dset.test_loader
        model.eval()
        with torch.no_grad():
            print('*'*80)
            print('Starting eval')
            correct, count = 0.0, 0.0
            pbar = tqdm(test_loader)
            for batch in pbar:
                batch = maybe_dictionarize_batch(batch)
                inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                count += len(logits)
                pbar.set_description(
                    f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")
            top1 = correct / count
        print(f'Val acc at epoch {epoch}: {100*top1:.2f}')

        model_path = os.path.join(args.model_location, f'{args.name}_{epoch + 1}.pt')
        print('Saving model to', model_path)
        torch.save(model.module.state_dict(), model_path)

