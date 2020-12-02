
## [MIX'EM: Unsupervised Image Classification using a Mixture of Embeddings](https://arxiv.org/abs/2007.09502) (ACCV 2020)

![](mixem_arch.png)

**Abstract:** We present MIX'EM, a novel solution for unsupervised image classification. MIX'EM generates representations that by themselves are sufficient to drive a general-purpose clustering algorithm to deliver high-quality classification. This is accomplished by building a mixture of embeddings module into a contrastive visual representation learning framework in order to disentangle representations at the category level. It first generates a set of embedding and mixing coefficients from a given visual representation, and then combines them into a single embedding. We introduce three techniques to successfully train MIX'EM and avoid degenerate solutions; (i) diversify mixture components by maximizing entropy, (ii) minimize instance conditioned component entropy to enforce a clustered embedding space, and (iii) use an associative embedding loss to enforce semantic separability. By applying (i) and (ii), semantic categories emerge through the mixture coefficients, making it possible to apply (iii). Subsequently, we run K-means on the representations to acquire semantic classification. We conduct extensive experiments and analyses on STL10, CIFAR10, and CIFAR100-20 datasets, achieving state-of-the-art classification accuracy of 78\%, 82\%, and 44\%, respectively. To achieve robust and high accuracy, it is essential to use the mixture components to initialize K-means. Finally, we report competitive baselines (70\% on STL10) obtained by applying K-means to the "normalized" representations learned using the contrastive loss.

### Training

First, use the `environment.yml` file to create an Anaconda environment with all dependencies. Please note that we have used PyTorch 1.7 for developing our models. 

Training configuration should be specified at `config.yaml`. To make things more convenient, some of the arguments can be set both in `config.yaml` and via the command line. In that case, the command line will overwrite `config.yaml`

#### To train with random initialization
`python run.py --config_path config.yaml --log_dir /path/to/log/dir --dataroot /path/to/dataset/dir --lineardataroot /path/to/dataset/dir/for/lineareval`

#### To train with initialization form a pre-trained SimCLR encoder
`python run.py --config_path config.yaml --log_dir /path/to/log/dir --dataroot /path/to/dataset/dir --lineardataroot /path/to/dataset/dir/for/lineareval --init_from /path/to/SimC:R/enocer/checkpoint`

#### To resume training
`python run.py --config_path config.yaml --log_dir /path/to/existing/expe/dir --resume --dataroot /path/to/dataset/dir --lineardataroot /path/to/dataset/dir/for/lineareval --init_from /path/to/SimC:R/enocer/checkpoint`

#### To train plain SimCLR
Remove "Mixture" from the "model" in the `config.yaml` file.

### Clustering

To evaluate the clusters, use the `cluster_eval.py` script as follows:

`python cluster_eval.py <dataset_dir> <dataset_name> <model_path> <model_config_path> <num_clusters>`

### Correspondence
alivaramesh@gmail.com

#### Acknoledgements
In developing our codebase, we have used [the implementation of SimCLR by Thalles Silva](https://github.com/sthalles/SimCLR)
