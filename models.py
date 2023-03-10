import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm


######################################################################################################
############################## MASKED EPIC CLASSIFER #################################################
######################################################################################################


class EPiC_layer_mask(nn.Module):
    """Definition of the EPIC layer"""

    def __init__(
        self,
        local_in_dim: int,
        hid_dim: int,
        latent_dim: int,
        sum_scale: float = 1e-2,
    ):
        """Initialise EPiC layer

        Parameters
        ----------
        local_in_dim : int
            Dimension of local features
        hid_dim : int
            Dimension of hidden layer
        latent_dim : int
            Dimension of latent space
        sum_scale : float, optional
            Scale factor for the result of the sum pooling operation, by default 1e-2
        """
        super().__init__()
        self.fc_global1 = weight_norm(nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim))
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))
        self.sum_scale = sum_scale

    def forward(self, x_global, x_local, mask):
        """Definition of the EPiC layer forward pass

        Parameters
        ----------
        x_global : torch.tensor
            Global features of shape [batch_size, dim_latent_global]
        x_local : torch.tensor
            Local features of shape [batch_size, N_points, dim_latent_local]
        mask : torch.tensor
            Mask of shape [batch_size, N_points, 1]. All non-padded values are
            "True", padded values are "False".
            This allows to exclude zero-padded points from the sum/mean aggregation
            functions

        Returns
        -------
        x_global
            Global features after the EPiC layer transformation
        x_local
            Local features after the EPiC layer transformation
        """
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)  # get number of global features

        # calculate the mean along the axis that represents the sets
        # communication between points is masked
        x_pooled_mean = (x_local * mask).mean(1, keepdim=False)
        # calculate the sum pooling and scale with the factor "sum_scale"
        x_pooled_sum = (x_local * mask).sum(1, keepdim=False) * self.sum_scale
        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global], 1)
        # new intermediate step
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))
        # with residual connection before AF
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global)

        # point wise function does not need to be masked
        # first add dimension, than expand it
        x_global2local = x_global.view(-1, 1, latent_global).repeat(1, n_points, 1)
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        # with residual connection before AF
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


class EPiC_discriminator_mask(nn.Module):
    """EPiC classifier"""

    def __init__(self, args):
        """Initialise the EPiC classifier

        Parameters
        ----------
        args : keyword argruments
            Expects:
                hid_d = dimension of the hidden layers in the phi MLPs
                feats = number of local features
                epic_layers = number of epic layers
                latent = dimension of the latent space (in the networks that act
                         on the point clouds)
        """
        super().__init__()
        self.hid_d = args.hid_d
        self.feats = args.feats
        self.epic_layers = args.epic_layers
        self.latent = args.latent  # used for latent size of equiv concat
        self.sum_scale = args.sum_scale

        self.fc_l1 = weight_norm(nn.Linear(self.feats, self.hid_d))
        self.fc_l2 = weight_norm(nn.Linear(self.hid_d, self.hid_d))

        self.fc_g1 = weight_norm(nn.Linear(int(2 * self.hid_d), self.hid_d))
        self.fc_g2 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.epic_layers):
            self.nn_list.append(
                EPiC_layer_mask(
                    self.hid_d, self.hid_d, self.latent, sum_scale=self.sum_scale
                )
            )

        self.fc_g3 = weight_norm(
            nn.Linear(int(2 * self.hid_d + self.latent), self.hid_d)
        )
        self.fc_g4 = weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.fc_g5 = weight_norm(nn.Linear(self.hid_d, 1))

    def forward(self, x, mask):
        """Forward propagation through the network

        Parameters
        ----------
        x : torch.tensor
            Input tensor of shape [batch_size, N_points, N_features]
        mask : torch.tensor
            Mask of shape [batch_size, N_points, 1]
            This allows to exclude zero-padded points from the sum/mean aggregation
            functions

        Returns
        -------
        x
            Output of the network
        """
        # local encoding
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)

        # global features: masked
        # mean over points dim.
        x_mean = (x_local * mask).mean(1, keepdim=False)
        # sum over points dim.
        x_sum = (x_local * mask).sum(1, keepdim=False) * self.sum_scale
        x_global = torch.cat([x_mean, x_sum], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.epic_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, mask)

        # again masking global features
        # mean over points dim.
        x_mean = (x_local * mask).mean(1, keepdim=False)
        # sum over points dim.
        x_sum = (x_local * mask).sum(1, keepdim=False) * self.sum_scale
        x = torch.cat([x_mean, x_sum, x_global], 1)

        x = F.leaky_relu(self.fc_g3(x))
        x = F.leaky_relu(self.fc_g4(x) + x)
        x = self.fc_g5(x)
        return x


######################################################################################################
############################## CONDITIONAL MASKED EPIC CLASSIFER #####################################
######################################################################################################


# EPiC layer
class EPiC_layer_cond_mask(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim, cond_feats=1, sum_scale=1e-2):
        super().__init__()
        self.fc_global1 = weight_norm(
            nn.Linear(int(2 * hid_dim) + latent_dim + cond_feats, hid_dim)
        )
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))
        self.sum_scale = sum_scale

    def forward(self, x_global, x_local, cond_tensor, mask):  # shapes:
        # - x_global[b,latent]
        # - x_local[b,n,latent_local]
        # - points_tensor [b,cond_feats]
        # - mask[B,N,1]
        # mask: all non-padded values = True      all zero padded = False
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        # communication between points is masked
        x_pooled_mean = (x_local * mask).mean(1, keepdim=False)
        x_pooled_sum = (x_local * mask).sum(1, keepdim=False) * self.sum_scale
        x_pooledCATglobal = torch.cat(
            [x_pooled_mean, x_pooled_sum, x_global, cond_tensor], 1
        )
        # new intermediate step
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))
        # with residual connection before AF
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global)

        # point wise function does not need to be masked
        # first add dimension, than expand it
        x_global2local = x_global.view(-1, 1, latent_global).repeat(1, n_points, 1)
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        # with residual connection before AF
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


# EPIC classifer
class EPiC_discriminator_cond_mask(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hid_d = args["hid_d"]
        self.feats = args["feats"]
        self.equiv_layers = args["equiv_layers_discriminator"]
        self.latent = args["latent"]  # used for latent size of equiv concat
        self.cond_feats = args["cond_feats"]
        self.sum_scale = args["sum_scale"]

        self.fc_l1 = self.weight_norm(nn.Linear(self.feats, self.hid_d))
        self.fc_l2 = self.weight_norm(nn.Linear(self.hid_d, self.hid_d))

        self.fc_g1 = self.weight_norm(
            nn.Linear(int(2 * self.hid_d + self.cond_feats), self.hid_d)
        )
        self.fc_g2 = self.weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(
                EPiC_layer_cond_mask(
                    self.hid_d,
                    self.hid_d,
                    self.latent,
                    self.cond_feats,
                    sum_scale=self.sum_scale,
                )
            )

        self.fc_g3 = self.weight_norm(
            nn.Linear(int(2 * self.hid_d + self.latent + self.cond_feats), self.hid_d)
        )
        self.fc_g4 = self.weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.fc_g5 = self.weight_norm(nn.Linear(self.hid_d, 1))

    def forward(self, x, cond_tensor, mask):
        # x [B,N,F]    cond_tensor B,C     mask B,N,1
        # local encoding
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)

        # global features: masked
        x_mean = (x_local * mask).mean(1, keepdim=False)  # mean over points dim.
        # mean over points dim.
        x_sum = (x_local * mask).sum(1, keepdim=False) * self.sum_scale
        x_global = torch.cat([x_mean, x_sum, cond_tensor], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, cond_tensor, mask)

        # again masking global features
        x_mean = (x_local * mask).mean(1, keepdim=False)  # mean over points dim.
        # sum over points dim.
        x_sum = (x_local * mask).sum(1, keepdim=False) * self.sum_scale
        x = torch.cat([x_mean, x_sum, x_global, cond_tensor], 1)

        x = F.leaky_relu(self.fc_g3(x))
        x = F.leaky_relu(self.fc_g4(x) + x)
        x = self.fc_g5(x)
        return x
