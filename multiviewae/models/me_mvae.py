import torch
import hydra
from ..base.constants import MODEL_MEMVAE
from ..base.base_model import BaseModelVAE
from ..base.representations import ProductOfExperts, MeanRepresentation

class me_mVAE(BaseModelVAE):
    r"""
    Multimodal Variational Autoencoder (MVAE).

    Loss optimises the ELBO term from the joint posterior distribution, as well as the separate ELBO terms for each view.
    me_mVAE stands for multi ELBO Multimodal VAE

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.beta (int, float): KL divergence weighting term.
            - model.join_type (str): Method of combining encoding distributions.
            - model.warmup (int): KL term weighted by beta linearly increased to 1 over this many epochs.
            - model.use_prior (bool): Whether to use a prior expert when combining encoding distributions.
            - model.sparse (bool): Whether to enforce sparsity of the encoding distribution.
            - model.threshold (float): Dropout threshold applied to the latent dimensions. Default is 0.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
    
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Wu, M., & Goodman, N.D. (2018). Multimodal Generative Models for Scalable Weakly-Supervised Learning. NeurIPS.

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_MEMVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.join_type = self.cfg.model.join_type
        if self.join_type == "PoE":
            self.join_z = ProductOfExperts()
        elif self.join_type == "Mean":
            self.join_z = MeanRepresentation()
        if self.warmup is not None:
            self.beta_vals = torch.linspace(0, self.beta, self.warmup)

    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): Single element list of joint encoding distribution.
            (list): List containing separate encoding distributions.
        """
        mu = []
        logvar = []
        qz_xs = []
        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_)
            logvar.append(logvar_)
            qz_x = hydra.utils.instantiate( 
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_, logvar=logvar_
            )
            qz_xs.append(qz_x)
        if not self.sparse and self.use_prior:
            #get mu and logvar from prior expert
            mu_ = self.prior.mean
            mu_ = mu_.expand(mu[0].shape).to(mu[0].device)            
            logvar_ = torch.log(self.prior.variance).to(mu[0].device)
            logvar_ = logvar_.expand(logvar[0].shape)              
            mu.append(mu_)
            logvar.append(logvar_)
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        mu_out, logvar_out = self.join_z(mu, logvar)
        qz_x = hydra.utils.instantiate( 
            self.cfg.encoder.default.enc_dist, loc=mu_out, logvar=logvar_out
        )
        if self._training:
            return [qz_x], qz_xs
        return [qz_x]

    def encode_subset(self, x, subset):
        r"""Forward pass through encoder networks for the specified subset.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): Single element list of joint encoding distribution.
        """
        mu = []
        logvar = []
        for i in subset:
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_)
            logvar.append(logvar_)
        if not self.sparse and self.use_prior:
            #get mu and logvar from prior expert
            mu_ = self.prior.mean
            mu_ = mu_.expand(mu[0].shape).to(mu[0].device)           
            logvar_ = torch.log(self.prior.variance).to(mu[0].device)     
            logvar_ = logvar_.expand(logvar[0].shape)      
            mu.append(mu_)
            logvar.append(logvar_)
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        mu_out, logvar_out = self.join_z(mu, logvar)
    
        qz_x = hydra.utils.instantiate(
            self.cfg.encoder.default.enc_dist, loc=mu_out, logvar=logvar_out
        )
        return [qz_x] 
           
           
    def decode(self, qz_x):
        r"""Forward pass of joint latent dimensions through decoder networks.
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """      
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def decode_subset(self, qz_x, subset):
        r"""Forward pass of joint latent dimensions through decoder networks for the specified subset.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """  
        px_zs = []
        for i in subset:
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]
    
    def decode_separate(self, qz_xs):
        r"""Forward pass of each view specific latent dimensions through the respective decoder network.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the view specific latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """  
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_xs[i]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        r"""Apply encode, decode, encode_separate and decode_separate methods to input data to generate the joint and modality specific latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qz_x, qz_xs = self.encode(x)
        px_zs = self.decode(qz_x)
        px_zss = self.decode_separate(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "px_zss": px_zss, "qz_x": qz_x, "qz_xs": qz_xs}
        return fwd_rtn

    def calc_kl(self, qz_xs):
        r"""Calculate KL-divergence loss.

        Args:
            qz_xs (list): list of encoding distributions.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = 0
        for i in range(len(qz_xs)):
            if self.sparse:
                kl += qz_xs[i].sparse_kl_divergence().mean(0).sum()
            else:
                kl += qz_xs[i].kl_divergence(self.prior).mean(0).sum()
        return kl

    def calc_ll(self, x, px_zs):
        r"""Calculate log-likelihood loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            px_zs (list): list of decoding distributions.

        Returns:
            ll (torch.Tensor): Log-likelihood loss.
        """
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[0][i].log_likelihood(x[i]).mean(0).sum() #first index is latent, second index is view
        return ll

    def loss_function(self, x, fwd_rtn):
        r"""Calculate multi ELBO Multimodal VAE loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the MVAE loss.
        """
        px_zs = fwd_rtn["px_zs"]
        qz_x = fwd_rtn["qz_x"]
        px_zss = fwd_rtn["px_zss"]
        qz_xs = fwd_rtn["qz_xs"]

        kl = self.calc_kl(qz_x)
        kl_separate = self.calc_kl(qz_xs)
        ll = self.calc_ll(x, px_zs)
        ll_separate = self.calc_ll(x, px_zss)

        if self.current_epoch > self.warmup - 1:
            total = (self.beta*kl  - ll )/self.n_views + self.beta*kl_separate - ll_separate
        else:
            total = (self.beta_vals[self.current_epoch]*kl  - ll )/self.n_views + self.beta_vals[self.current_epoch]*kl_separate - ll_separate
        
        losses = {"loss": total, "kl": kl, "ll": ll, "ll_separate": ll_separate, "kl_separate": kl_separate}

        return losses
