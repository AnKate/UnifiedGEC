from torch import nn
import os
import sys

fairseq_path = os.path.abspath(os.path.join(os.getcwd(), "gectoolkit", "model",
                                            "SynGEC", "src", "src_syngec", "fairseq2"))
sys.path.insert(0, fairseq_path)

from fairseq import utils


class SynGEC(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        self.device = config["device"]
        self.use_gpu = config['use_gpu']
        self.model = dataloader.model
        self.criterion = dataloader.criterion

        self.pretrained_tokenzier = dataloader.pretrained_tokenizer
        self.padding_id = self.pretrained_tokenzier.src_dict.index('<pad>')

    def forward(self, batch, dataloader):
        self.model.train()
        self.criterion.train()
        batch = utils.move_to_cuda(batch)
        loss, sample_size, logging_output = self.criterion(self.model, batch)
        tgt_tokens = batch['target']          # (batch_size,length)
        output = logging_output['net_output']
        y_pred_tokens = output[0].argmax(dim=-1)  # (batch_size,length)

        loss_dict = {"decode_result": output[0], "loss": loss, "sample_size": sample_size}
        return loss_dict

    def model_test(self, batch, dataloader):
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        tgt_tokens = batch['target']
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)
        src_outcoming_arc_mask, src_incoming_arc_mask, src_dpd_matrix, src_probs_matrix, src_nt, src_nt_lengths = None, None, None, None, None, None
        prev_output_tokens = batch["net_input"]["prev_output_tokens"]

        if "src_incoming_arc_mask" in batch["net_input"].keys():
            src_incoming_arc_mask = batch["net_input"]["src_incoming_arc_mask"]
        if "src_outcoming_arc_mask" in batch["net_input"].keys():
            src_outcoming_arc_mask = batch["net_input"]["src_outcoming_arc_mask"]
        if "src_dpd_matrix" in batch["net_input"].keys():
            src_dpd_matrix = batch["net_input"]["src_dpd_matrix"]
        if "src_probs_matrix" in batch["net_input"].keys():
            src_probs_matrix = batch["net_input"]["src_probs_matrix"]
        if "source_tokens_nt" in batch["net_input"].keys():
            src_nt = batch["net_input"]["source_tokens_nt"]
        if "source_tokens_nt_lengths" in batch["net_input"].keys():
            src_nt_lengths = batch["net_input"]["source_tokens_nt_lengths"]

        return_all_hiddens = True
        features_only = False
        alignment_layer = None
        alignment_heads = None

        if self.use_gpu:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
            prev_output_tokens = prev_output_tokens.cuda()
            if constraints is not None:
                constraints = constraints.cuda()
            if src_incoming_arc_mask is not None:
                for i in range(len(src_incoming_arc_mask)):
                    src_incoming_arc_mask[i] = src_incoming_arc_mask[i].cuda()
            if src_outcoming_arc_mask is not None:
                for i in range(len(src_outcoming_arc_mask)):
                    src_outcoming_arc_mask[i] = src_outcoming_arc_mask[i].cuda()
            if src_dpd_matrix is not None:
                for i in range(len(src_dpd_matrix)):
                    src_dpd_matrix[i] = src_dpd_matrix[i].cuda()
            if src_probs_matrix is not None:
                for i in range(len(src_probs_matrix)):
                    src_probs_matrix[i] = src_probs_matrix[i].cuda()
            if src_nt is not None:
                src_nt = src_nt.cuda()
                src_nt_lengths = src_nt_lengths.cuda()

        encoder_out = self.model.encoder(
            src_tokens, src_lengths=src_lengths, src_outcoming_arc_mask=src_outcoming_arc_mask,
            src_incoming_arc_mask=src_incoming_arc_mask, src_dpd_matrix=src_dpd_matrix,
            src_probs_matrix=src_probs_matrix, source_tokens_nt=src_nt,
            source_tokens_nt_lengths=src_nt_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.model.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,  # false
            alignment_layer=alignment_layer,  # none
            alignment_heads=alignment_heads,  # none
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        y_pred = decoder_out[0]
        predict_ids = y_pred.argmax(dim=-1)

        return predict_ids,tgt_tokens

