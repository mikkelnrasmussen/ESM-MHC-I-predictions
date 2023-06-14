import torch
import esm

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

RBD=[["RBD", "MKGIDNTAYSYIDDLTCCTRVIMADYLNSDYRYNKDVDLDLVKLFLENGKPHGIMCSIVPLWRNDKETIFLILKTMNSDVLQHILIEYMTFGDIPLVEYGTVVNKEAIHGYFRNINIDSYTMKYLLKKEGGDAVNHLDDGEIPIGHLCKSNYECYNFYTYTYKKGLCDMSYACPILSTINICLPYLKDINMIDKRGETLLHKAVRYNKQSLVSLLLESGSDVNIRSNNGYTCIAIAINESRNIELLKMLLCHKPTLDYVIDSLREISNIVDNYYAIKQCIKYAMIIDDCTSSKIPEFISQRYNDYIDLCN"]]

batch_labels, batch_strs, batch_tokens = batch_converter(RBD)
print(len(batch_strs[0]))
print(batch_tokens)

with torch.no_grad():
  results = model(batch_tokens[0:1,:], repr_layers=[6], return_contacts=False)
token_representations = results["representations"][6]
print(token_representations.shape)