# Protein Structure Contact Map Predictor using ESM-2

An end-to-end Deep Learning and data engineering pipeline built in PyTorch that leverages Meta's state-of-the-art ESM-2 (Evolutionary Scale Modeling) transformer embeddings to predict the 3D spatial contact maps of proteins directly from raw amino acid sequences.

# How I Built a Protein 3D Contact Predictor (And What Broke Along the Way)

A lot of machine learning tutorials give you perfectly clean data where the model magically works on the first try. This isn't one of those projects. 

This is the story of how I built an end-to-end deep learning pipeline in PyTorch to predict the 3D shape of a protein from nothing but its raw sequence string. I walked into this project wanting to understand how structural biology and transformers intersect, and I walked out with a battle-tested appreciation for data engineering, vanishing gradients, and extreme class imbalances.

Here is a look at the journey, the roadblocks, and how I got the network to pull off a 100% recall rate on hidden test data.

---

## 🧬 The Big Picture Idea

If you have a flat string of amino acids, how do you guess its 3D shape? One powerful way is to predict its **Contact Map**—a grid that looks at every possible pair of amino acids in the chain and marks a `1` if they physically touch each other in 3D space, or a `0` if they float apart. 

To do this, I grabbed Meta's pre-trained **ESM-2 language model** (`esm2_t6_8M_UR50D`) to convert the biological text into high-dimensional evolutionary vectors ($L \times 320$). Then, I built a custom neural network to compare every residue pair and map out that $L \times L$ grid.

---

## 🚧 Roadblocks & Breakthroughs

### 1. The Shape Mismatch (Messy Biology Data)
My first wall was data preprocessing. When I passed a protein sequence through ESM-2, it gave me a clean set of feature tokens. But when I downloaded the corresponding physical 3D file from the Protein Data Bank (PDB) to create my ground-truth contact map, the dimensions didn't match. For example, the text had 41 positions, but the structural file only had 37.

* **The Reality:** Real-world laboratory x-rays often miss physical coordinates for highly flexible parts of a protein. 
* **The Fix:** I wrote a custom tensor-slicing logic to dynamically force-align the genomic features ($X$) with the true structural grid ($Y$) before passing them into the network.

### 2. The Saturation Trap (Debugging All Zeros)
During my first multi-protein training run, the model completely choked. The output matrix was completely blank. It was predicting a flat zero for every single coordinate, giving me an absolute **0.00% Precision and Recall**. 

* **The Reality:** I was using PyTorch's `BCEWithLogitsLoss`. What I didn't realize was that this loss function secretly applies a Sigmoid activation under the hood for mathematical stability. Because I had *also* manually placed a `nn.Sigmoid()` layer at the very end of my custom network, the data was getting crushed by a Sigmoid twice. This completely froze my gradients and stopped the model from learning.
* **The Fix:** I stripped the manual Sigmoid layer out of my architecture. The raw numerical scores (logits) flowed cleanly into the loss function, and the network started learning instantly.

### 3. Beating the 98% Laziness Problem
In structural biology, data is incredibly sparse. Roughly 98% of a protein is just empty space, and only about 2% constitutes true physical contacts. A standard AI model is inherently lazy; it quickly realized that if it just guessed "No Contact" for every single pixel, it could achieve a massive 98% accuracy score without doing any real work.

* **The Reality:** Traditional accuracy metrics are a lie when data is this imbalanced. 
* **The Fix:** I injected a heavy penalty factor into the system (`pos_weight=25.0`). I told the optimizer: *"Missing an actual contact point will penalize your score 25 times harder than miscalculating empty space."* This forced the network to stop cheating and actively hunt for structural bonds.

### 4. Bypassing RAM Crashes (Disk Caching)
When I scaled up the project to handle a larger batch of 521 sequences, Google Colab's active memory immediately hit a ceiling and crashed. The massive high-dimensional transformer embeddings were simply too heavy to hold in active RAM simultaneously.

* **The Reality:** To build scalable AI pipelines, you have to decouple your steps.
* **The Fix:** I split the code. I built an automated script to handle the heavy lifting *once*—downloading structures and extracting ESM-2 features—and immediately cached them onto disk as independent `.pt` files. Then, I wrote a custom PyTorch `Dataset` and `DataLoader` to dynamically stream those files from disk one by one during training, dropping volatile RAM utilization to near zero.

---

## 📊 The Final Verification

To see if the model actually learned the laws of biology or just memorized its homework, I locked a test protein (`1bbl`) in a vault and only showed it to the model *after* training was completely finished. 

Here are the final metrics on that completely hidden protein:

```text
--- FINAL DATASET VALIDATION METRICS ---
True Contacts Pinpointed: 309
False Alarms Triggered:  1060
Precision Score:          0.2257
Recall Score:             1.0000
