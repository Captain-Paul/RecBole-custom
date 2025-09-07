# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        # LambdaLoss related
        self.candidate_size = config["candidate_size"] if "candidate_size" in config else 15
        self.lambda_temp = config["lambda_temp"] if "lambda_temp" in config else 0.1
        self.ndcg_cutoff = config["ndcg_cutoff"] if "ndcg_cutoff" in config else 10

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "LambdaLoss":
            pass
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def compute_lambda_weights(self, relevance_scores, ranking_scores):
        """
        计算Lambda权重，核心是模拟排序变化对NDCG的影响
        Args:
            relevance_scores: [batch_size, list_size] 相关性标签 (0或1)
            ranking_scores: [batch_size, list_size] 模型预测得分
        Returns:
            lambda_weights: [batch_size, list_size, list_size] 权重矩阵
        """
        batch_size, list_size = relevance_scores.shape
        device = relevance_scores.device
        
        # 计算理想DCG (IDCG)
        sorted_relevance, _ = torch.sort(relevance_scores, dim=1, descending=True)
        position_discounts = torch.log2(torch.arange(2, list_size + 2, device=device).float())
        idcg = torch.sum(sorted_relevance / position_discounts.unsqueeze(0), dim=1, keepdim=True)
        idcg = torch.clamp(idcg, min=1e-8)  # 避免除零
        
        # 当前排序的DCG
        _, sorted_indices = torch.sort(ranking_scores, dim=1, descending=True)
        current_dcg = torch.zeros(batch_size, 1, device=device)
        for b in range(batch_size):
            sorted_rel = relevance_scores[b, sorted_indices[b]]
            current_dcg[b] = torch.sum(sorted_rel / position_discounts)
        
        # 计算Lambda权重矩阵
        lambda_matrix = torch.zeros(batch_size, list_size, list_size, device=device)
        
        for i in range(list_size):
            for j in range(list_size):
                if i != j:
                    # 模拟交换位置i和j后的DCG变化
                    delta_dcg = self._compute_dcg_delta(
                        relevance_scores, sorted_indices, i, j, position_discounts
                    )
                    
                    # 归一化的NDCG变化
                    delta_ndcg = torch.abs(delta_dcg) / idcg
                    
                    # Lambda权重 = |ΔNDCG|
                    lambda_matrix[:, i, j] = delta_ndcg.squeeze()
        
        return lambda_matrix
    
    def _compute_dcg_delta(self, relevance, sorted_indices, pos_i, pos_j, discounts):
        """计算交换两个位置后DCG的变化量"""
        batch_size = relevance.shape[0]
        delta_dcg = torch.zeros(batch_size, 1, device=relevance.device)
        
        for b in range(batch_size):
            rel_i = relevance[b, sorted_indices[b, pos_i]]
            rel_j = relevance[b, sorted_indices[b, pos_j]]
            
            # 交换前后的贡献差异
            old_contrib = rel_i / discounts[pos_i] + rel_j / discounts[pos_j]
            new_contrib = rel_j / discounts[pos_i] + rel_i / discounts[pos_j]
            
            delta_dcg[b] = new_contrib - old_contrib
            
        return delta_dcg
    
    def lambda_loss_computation(self, user_embeddings, pos_items, neg_item_candidates):
        """
        LambdaLoss的核心计算逻辑
        Args:
            user_embeddings: [batch_size, hidden_size] 用户序列表示
            pos_items: [batch_size] 正样本物品ID
            neg_item_candidates: [batch_size, neg_size] 负样本候选物品ID
        """
        batch_size = user_embeddings.shape[0]
        
        # 构建完整候选集合：正样本 + 负样本
        all_items = torch.cat([
            pos_items.unsqueeze(1),  # [batch_size, 1]
            neg_item_candidates      # [batch_size, neg_size]
        ], dim=1)  # [batch_size, 1 + neg_size]
        
        # 获取所有候选物品的嵌入
        all_item_embeddings = self.item_embedding(all_items)  # [batch_size, list_size, hidden_size]
        
        # 计算相似度得分
        similarity_scores = torch.bmm(
            all_item_embeddings,
            user_embeddings.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, list_size]
        
        # 构建相关性标签：正样本为1，负样本为0
        relevance_labels = torch.zeros_like(similarity_scores)
        relevance_labels[:, 0] = 1.0  # 第一个位置是正样本
        
        # 计算Lambda权重
        lambda_weights = self.compute_lambda_weights(relevance_labels, similarity_scores)
        
        # 计算排序损失
        # 对于每对物品(i,j)，如果物品i比物品j更相关，但得分更低，则产生损失
        score_diff = similarity_scores.unsqueeze(2) - similarity_scores.unsqueeze(1)  # [batch_size, list_size, list_size]
        relevance_diff = relevance_labels.unsqueeze(2) - relevance_labels.unsqueeze(1)  # [batch_size, list_size, list_size]
        
        # 只考虑相关性不同的物品对
        valid_pairs = (relevance_diff > 0).float()
        
        # 使用sigmoid函数计算排序概率
        ranking_probs = torch.sigmoid(score_diff / self.lambda_temp)
        
        # LambdaLoss = Σ λ(i,j) * log(1 + exp(-(s_i - s_j)/T))
        # 等价于 λ(i,j) * log(1 + exp(-score_diff/T))
        pairwise_loss = -torch.log(ranking_probs + 1e-8)
        
        # 应用Lambda权重和有效对掩码
        weighted_loss = lambda_weights * pairwise_loss * valid_pairs
        
        # 平均损失
        total_loss = weighted_loss.sum() / (valid_pairs.sum() + 1e-8)
        
        return total_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss
        else:
            neg_items = interaction[self.NEG_ITEM_ID]
            if neg_items.dim() == 1:
                neg_items = neg_items.unsqueeze(1)
            loss = self.lambda_loss_computation(seq_output, pos_items, neg_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores