# autoencoders
implementations of various molecular autoencoders

## organization

the repo is organized along the following logic:
1. molecules can be represented as sequences, where the sequence is can be characters (i.e., SMILES) or otherwise (i.e., a character or graph grammar)
2. Given (1), then all autoencoders share the same base [`Encoder`](./autoencoders/modules/encoder.py)
3. What differentiates VAEs (e.g., character-, grammar-, or graph-VAE) is then actually the _decoder_. As such, there are submodules corresponding to each type of VAE implemented (currently only character and grammar)

## ELI5

### Character VAE (CVAE)

- molecules are represented as SMILES strings
- SMILES strings are (i) tokenized then (ii) each token is encoded to an integer based on a supplied vocabulary. There are fairly standard SMILES vocabularies and tokenization schemes taken from [Schwaller et al.](https://github.com/rxn4chemistry/rxnfp) that we use here, but you can define your own. A vocabulary consists of all the tokens we expected to see in a SMILES string (e.g., "C", "Br", "O", etc.) as well as some "special tokens": `SOS`, `EOS`, `PAD`, and `UNK` ("start of sequence", "end of sequence", "pad character", and "unknown", respectively). The `PAD` token helps square off a jagged list of sequences (i.e., make a list of sequences with unequal lengths all the same length). The `UNK` token covers cases where an input sequence has a token that you didn't anticipate (e.g., "Pd") without breaking things.
- Given a sequence of encoded tokens, we use an `Encoder` to embed this sequence into a hidden vector: `h`.
- We feed this hidden vector to two _independent_ linear layers that map this hidden vector to both the mean and logvar ("log of the variance") of a distribution in latent space: `z_mean` and `z_logvar`
- `z_mean` and `z_logvar` define a distribution in latent space, so we then sample from this distibution to get a single latent vector `z`. This is where the term "variational" comes in. If we had either (i) defined only a single linear layer in the step above or equivalently tossed out `z_logvar` and only used `z_mean`, we would be left with a plain autoencoder (not technically true)
- At generation time, we start with some latent vector `z` and the `SOS` ("start of sequence") token which we feed into our `CharacterDecoder`. This outputs (1) an unnormalized probability distribution over all tokens (including our special tokens), from which we sample to get the next token in the sequence, and (2) an updated hidden state.
- We feed in the next token and updated hidden state iteratively until we either hit some maximum number of iterations or sample the `EOS` token.
- This output sequence of encoded tokens can then be decoded back to a sequence of tokens from which we again have a SMILES string!

#### Notes:
- While it guaranteed that our sequence starts with the `SOS` token and _likely_ that it ends with the `EOS` token (subject to sampling that before we run out of iterations), it is _technically_ possible that we have sampled other special tokens in the middle of our sequence (i.e., our sequence could very well have `SOS`, `PAD`, and `UNK` in the middle). **However**, the VAE quickly learns to avoid this "naughty" behavior
- Even fixing the above problem, the CVAE has no "innate" concept of syntactic validity for SMILES strings. We know that if we open an aromatic ring in a SMILES string (e.g., "c1") we must eventually close it, but the CVAE has no guarantee that it will. Formally, this is called a "grammar". **However** (again), it turns out that the grammar of SMILES strings isn't that hard to learn, and most CVAEs with a large enough corpus will learn to avoid this behavior.